import os
from typing import Optional

import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv


def get_db_config(include_database: bool = False) -> dict:
    """
    Read MySQL credentials from .env file / environment variables.

    include_database=False:
        used when creating the database itself

    include_database=True:
        used when connecting to the target database
    """
    load_dotenv()

    config = {
        "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
        "port": int(os.getenv("MYSQL_PORT", 3306)),
        "user": os.getenv("MYSQL_USER", "history_user"),
        "password": os.getenv("MYSQL_PASSWORD", "StrongPassword123!"),
        "autocommit": False,
    }

    if include_database:
        config["database"] = os.getenv("MYSQL_DATABASE", "history")

    return config


def get_connection(include_database: bool = True):
    """
    Create a MySQL connection.
    """
    config = get_db_config(include_database=include_database)
    return mysql.connector.connect(**config)


def create_database(conn, db_name: str) -> None:
    """
    Create database if it does not exist.
    """
    sql = f"""
    CREATE DATABASE IF NOT EXISTS `{db_name}`
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci
    """
    cursor = conn.cursor()
    cursor.execute(sql)
    cursor.close()
    print(f"Database `{db_name}` checked/created.")


def create_news_table(conn) -> None:
    """
    Create the news table.
    """
    sql = """
    CREATE TABLE IF NOT EXISTS news (
        id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
        title VARCHAR(1000) NOT NULL,
        url TEXT NOT NULL,
        url_hash CHAR(64) NOT NULL,
        source VARCHAR(255) DEFAULT NULL,
        published_at DATETIME DEFAULT NULL,
        fetched_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        language VARCHAR(32) DEFAULT NULL,
        country_code VARCHAR(16) DEFAULT NULL,
        keyword VARCHAR(255) DEFAULT NULL,
        summary TEXT DEFAULT NULL,
        content LONGTEXT DEFAULT NULL,
        sentiment_score DECIMAL(8,4) DEFAULT NULL,
        relevance_score DECIMAL(8,4) DEFAULT NULL,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (id),
        UNIQUE KEY uk_news_url_hash (url_hash),
        KEY idx_news_published_at (published_at),
        KEY idx_news_source (source),
        KEY idx_news_keyword (keyword)
    ) ENGINE=InnoDB
      DEFAULT CHARSET=utf8mb4
      COLLATE=utf8mb4_unicode_ci
    """
    cursor = conn.cursor()
    cursor.execute(sql)
    cursor.close()
    print("Table `news` checked/created.")


def create_performance_table(conn) -> None:
    """
    Create the performance table.
    """
    sql = """
    CREATE TABLE IF NOT EXISTS performance (
        id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
        run_id VARCHAR(128) NOT NULL,
        model_name VARCHAR(255) NOT NULL,
        task_name VARCHAR(255) DEFAULT NULL,
        metric_date DATE DEFAULT NULL,
        horizon VARCHAR(64) DEFAULT NULL,
        accuracy DECIMAL(10,6) DEFAULT NULL,
        precision_score DECIMAL(10,6) DEFAULT NULL,
        recall_score DECIMAL(10,6) DEFAULT NULL,
        f1_score DECIMAL(10,6) DEFAULT NULL,
        auc_score DECIMAL(10,6) DEFAULT NULL,
        mae DECIMAL(18,8) DEFAULT NULL,
        rmse DECIMAL(18,8) DEFAULT NULL,
        sharpe_ratio DECIMAL(18,8) DEFAULT NULL,
        max_drawdown DECIMAL(18,8) DEFAULT NULL,
        annual_return DECIMAL(18,8) DEFAULT NULL,
        information_ratio DECIMAL(18,8) DEFAULT NULL,
        notes TEXT DEFAULT NULL,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (id),
        KEY idx_perf_run_id (run_id),
        KEY idx_perf_model_name (model_name),
        KEY idx_perf_metric_date (metric_date),
        KEY idx_perf_task_name (task_name)
    ) ENGINE=InnoDB
      DEFAULT CHARSET=utf8mb4
      COLLATE=utf8mb4_unicode_ci
    """
    cursor = conn.cursor()
    cursor.execute(sql)
    cursor.close()
    print("Table `performance` checked/created.")


def init_database() -> None:
    """
    Full initialization flow:
    1. connect to MySQL server
    2. create database
    3. reconnect to target database
    4. create tables
    """
    db_name = os.getenv("MYSQL_DATABASE", "history")
    server_conn = None
    db_conn = None

    try:
        # Connect without selecting a database first
        server_conn = get_connection(include_database=False)
        print("Connected to MySQL server.")

        # Create database
        create_database(server_conn, db_name)

        # Reconnect using the target database
        db_conn = get_connection(include_database=True)
        print(f"Connected to database `{db_name}`.")

        # Create tables
        create_news_table(db_conn)
        create_performance_table(db_conn)

        print("Initialization completed successfully.")

    except Error as e:
        print(f"MySQL error: {e}")
        raise

    finally:
        if server_conn is not None and server_conn.is_connected():
            server_conn.close()
            print("Server connection closed.")

        if db_conn is not None and db_conn.is_connected():
            db_conn.close()
            print("Database connection closed.")


if __name__ == "__main__":
    init_database()