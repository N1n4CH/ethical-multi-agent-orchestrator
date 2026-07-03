import pandas as pd
import numpy as np

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent

from sqlalchemy import create_engine, inspect
from urllib.parse import quote_plus

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
)


class StructuredDataAgent:

    # ============================================
    # INIT
    # ============================================
    def __init__(
        self,
        azure_endpoint,
        api_key,
        deployment,
        db_config,
        api_version="2024-12-01-preview",
        temperature=0.7,
    ):
        encoded_password = quote_plus(db_config["password"])
        self.db_uri = (
            f"postgresql+psycopg://{db_config['user']}:{encoded_password}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
        )

        self.model = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            api_key=api_key,
            deployment_name=deployment,
            temperature=temperature,
        )

        self.db = SQLDatabase.from_uri(self.db_uri)
        toolkit = SQLDatabaseToolkit(db=self.db, llm=self.model)
        tools = toolkit.get_tools()

        self.agent_executor = create_react_agent(self.model, tools)

    # ============================================
    # PUBLIC METHOD
    # ============================================
    def ask(self, question, verbose=False, run_bias_audit=True):

        response_text = ""

        for chunk in self.agent_executor.stream(
            {"messages": [("user", question)]},
            stream_mode="values",
        ):
            msg = chunk["messages"][-1]
            response_text = msg.content or ""

        if run_bias_audit:
            print("\n" + "=" * 80)
            print("Using Strucutred Data stored in PostgresQL database to answer question")
            print("Performing a Fairness audit before providing answer")
            print("=" * 80)
            self.__auto_bias_check(self.db_uri)

        return {"question": question, "response": response_text}

    # ============================================
    # TABLE DISCOVERY
    # ============================================
    def __discover_tables(self, db_uri):

        engine = create_engine(db_uri)
        inspector = inspect(engine)

        tables = []
        for schema in inspector.get_schema_names():
            if schema in ["information_schema", "pg_catalog"]:
                continue

            for table in inspector.get_table_names(schema=schema):
                full_name = f"{schema}.{table}" if schema != "public" else table
                tables.append(full_name)

        return tables, engine

    # ============================================
    # DEMOGRAPHIC DETECTION
    # ============================================
    def __find_demographic_columns(self, columns):

        keywords = [
            "race", "ethnicity", "gender", "sex", "age", "income",
            "disability", "religion", "nationality", "marital_status",
            "veteran", "orientation", "language", "citizenship",
        ]

        return [
            col for col in columns
            if any(k in col.lower() for k in keywords)
        ]

    # ============================================
    # FAIRNESS ENGINE
    # ============================================
    def __run_fairness_analysis(self, df, table_name, demographic_column):

        print("\n" + "-" * 70)
        print(f"TABLE: {table_name}")
        print(f"DEMOGRAPHIC COLUMN: {demographic_column}")
        print("-" * 70)

        for col in df.columns:

            if col == demographic_column:
                continue

            temp = df[[demographic_column, col]].dropna()
            if temp.empty:
                continue

            print(f"\nAnalyzing column: {col}")

            unique_count = temp[col].nunique()

            # ============================================
            # BINARY → Fairlearn
            # ============================================
            if unique_count == 2:

                print("Type: Binary (classification fairness)")

                y = temp[col]
                sensitive = temp[demographic_column]

                metric_frame = MetricFrame(
                    metrics={"selection_rate": selection_rate},
                    y_true=y,
                    y_pred=y,
                    sensitive_features=sensitive,
                )

                print("\nSelection Rate by Group:")
                print(metric_frame.by_group)

                dp_diff = demographic_parity_difference(
                    y_true=y,
                    y_pred=y,
                    sensitive_features=sensitive,
                )

                print(f"Demographic Parity Difference: {dp_diff:.4f}")

                if abs(dp_diff) > 0.10:
                    print("Potential bias detected")
                else:
                    print("Within fairness threshold")

            # ============================================
            # NUMERIC → Mean Comparison
            # ============================================
            elif pd.api.types.is_numeric_dtype(temp[col]):

                print("Type: Numeric (distribution fairness)")

                means = temp.groupby(demographic_column)[col].mean()
                print("\nMean by group:")
                print(means)

                min_val = means.min()
                max_val = means.max()
                ratio = min_val / max_val if max_val > 0 else 0

                print(f"Mean ratio (min/max): {ratio:.3f}")

                if ratio < 0.80:
                    print("Large disparity detected")
                else:
                    print("No major disparity")

            # ============================================
            # SMALL CATEGORICAL → Distribution Compare
            # ============================================
            elif 2 < unique_count <= 10:

                print("Type: Categorical (distribution comparison)")

                dist = pd.crosstab(
                    temp[demographic_column],
                    temp[col],
                    normalize="index"
                ) * 100

                print("\nPercentage distribution by group:")
                print(dist.round(2))

            else:
                print("Skipping (too many categories)")

    # ============================================
    # AUTO BIAS CHECK
    # ============================================
    def __auto_bias_check(self, db_uri):

        tables, engine = self.__discover_tables(db_uri)

        for table_name in tables:

            df = pd.read_sql(f"SELECT * FROM {table_name}", engine)

            demographic_cols = self.__find_demographic_columns(df.columns)

            if not demographic_cols:
                continue

            for demo_col in demographic_cols:
                self.__run_fairness_analysis(df, table_name, demo_col)
