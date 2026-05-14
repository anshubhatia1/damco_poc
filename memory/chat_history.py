# import sqlite3
# from datetime import datetime

# DB_PATH = "memory/chat_history.db"


# class ChatHistory:

#     def __init__(self):
#         self._initialize_database()

#     def _initialize_database(self):
#         conn = sqlite3.connect(DB_PATH)
#         cursor = conn.cursor()

#         cursor.execute("""
#         CREATE TABLE IF NOT EXISTS messages (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             session_id TEXT NOT NULL,
#             role TEXT NOT NULL,
#             content TEXT NOT NULL,
#             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#         )
#         """)

#         conn.commit()
#         conn.close()

#     def add_message(self, session_id, role, content):
#         conn = sqlite3.connect(DB_PATH)
#         cursor = conn.cursor()

#         cursor.execute("""
#         INSERT INTO messages (session_id, role, content)
#         VALUES (?, ?, ?)
#         """, (session_id, role, content))

#         conn.commit()
#         conn.close()

#     def get_messages(self, session_id, limit=10):
#         conn = sqlite3.connect(DB_PATH)
#         cursor = conn.cursor()

#         cursor.execute("""
#         SELECT role, content
#         FROM messages
#         WHERE session_id = ?
#         ORDER BY id ASC
#         LIMIT ?
#         """, (session_id, limit))

#         rows = cursor.fetchall()

#         conn.close()

#         return rows

#     def clear_session(self, session_id):
#         conn = sqlite3.connect(DB_PATH)
#         cursor = conn.cursor()

#         cursor.execute("""
#         DELETE FROM messages
#         WHERE session_id = ?
#         """, (session_id,))

#         conn.commit()
#         conn.close()

import sqlite3
from contextlib import contextmanager

DB_PATH = "memory/chat_history.db"


class ChatHistory:

    def __init__(self):
        self._initialize_database()

    # =====================================================
    # Connection Manager
    # =====================================================

    @contextmanager
    def _get_connection(self):

        conn = sqlite3.connect(DB_PATH)

        try:
            yield conn

        finally:
            conn.close()

    # =====================================================
    # Initialize Database Tables
    # =====================================================

    def _initialize_database(self):

        with self._get_connection() as conn:

            cursor = conn.cursor()

            # -------------------------------------------------
            # Messages Table
            # -------------------------------------------------

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                session_id TEXT NOT NULL,

                role TEXT NOT NULL,

                content TEXT NOT NULL,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # -------------------------------------------------
            # Retrieval Trace Table
            # -------------------------------------------------

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_traces (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                session_id TEXT NOT NULL,

                query TEXT NOT NULL,

                chunk_id TEXT,

                document_title TEXT,

                page INTEGER,

                faiss_rank INTEGER,

                faiss_distance REAL,

                faiss_score REAL,

                rerank_rank INTEGER,

                rerank_score REAL,

                chunk_content TEXT,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # -------------------------------------------------
            # Usage Metrics Table
            # -------------------------------------------------

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS usage_metrics (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                session_id TEXT NOT NULL,

                query TEXT NOT NULL,

                prompt_tokens INTEGER,

                completion_tokens INTEGER,

                total_tokens INTEGER,

                total_cost REAL,

                latency_ms INTEGER,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            conn.commit()

    # =====================================================
    # Chat Message Persistence
    # =====================================================

    def add_message(
        self,
        session_id,
        role,
        content
    ):

        with self._get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute("""
            INSERT INTO messages (
                session_id,
                role,
                content
            )
            VALUES (?, ?, ?)
            """, (
                session_id,
                role,
                content
            ))

            conn.commit()

    def get_messages(
        self,
        session_id,
        limit=10
    ):

        with self._get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute("""
            SELECT role, content
            FROM messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """, (
                session_id,
                limit
            ))

            rows = cursor.fetchall()

            rows.reverse()

            return rows

    def clear_session(self, session_id):

        with self._get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute("""
            DELETE FROM messages
            WHERE session_id = ?
            """, (session_id,))

            conn.commit()

    # =====================================================
    # Retrieval Trace Persistence
    # =====================================================

    def log_retrieval_trace(
        self,
        session_id,
        query,
        retrieval_results
    ):

        with self._get_connection() as conn:

            cursor = conn.cursor()

            for item in retrieval_results:

                doc = item["doc"]

                metadata = doc.metadata

                cursor.execute("""
                INSERT INTO retrieval_traces (

                    session_id,
                    query,

                    chunk_id,

                    document_title,

                    page,

                    faiss_rank,

                    faiss_distance,

                    faiss_score,

                    rerank_rank,

                    rerank_score,

                    chunk_content

                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (

                    session_id,

                    query,

                    metadata.get("chunk_id"),

                    metadata.get(
                        "title",
                        metadata.get("book_title")
                    ),

                    metadata.get("page"),

                    item.get("faiss_rank"),

                    item.get("faiss_distance"),

                    item.get("faiss_score"),

                    item.get("rerank_rank"),

                    item.get("rerank_score"),

                    doc.page_content
                ))

            conn.commit()

    # =====================================================
    # Usage Metrics Persistence
    # =====================================================

    def log_usage_metrics(
        self,
        session_id,
        query,

        prompt_tokens,
        completion_tokens,
        total_tokens,

        total_cost,

        latency_ms,
    ):

        with self._get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute("""
            INSERT INTO usage_metrics (

                session_id,

                query,

                prompt_tokens,

                completion_tokens,

                total_tokens,

                total_cost,

                latency_ms

            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (

                session_id,

                query,

                prompt_tokens,

                completion_tokens,

                total_tokens,

                total_cost,

                latency_ms,
            ))

            conn.commit()

    def get_all_sessions(self):
        """
        Return all distinct sessions ordered by most recent activity.
        Each row: (session_id, first_user_question, last_created_at)
        Used by the frontend to render the conversation thread list.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    m.session_id,
                    first_q.content   AS title,
                    MAX(m.created_at) AS last_active
                FROM messages m
                INNER JOIN (
                    SELECT session_id, content
                    FROM messages
                    WHERE role = 'user'
                      AND id IN (
                          SELECT MIN(id) FROM messages
                          WHERE role = 'user'
                          GROUP BY session_id
                      )
                ) first_q ON m.session_id = first_q.session_id
                GROUP BY m.session_id
                ORDER BY last_active DESC
            """)
            return cursor.fetchall()   # list of (session_id, title, last_active)

    # =====================================================
    # Analytics Helpers
    # =====================================================

    def get_session_message_count(
        self,
        session_id
    ):

        with self._get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute("""
            SELECT COUNT(*)
            FROM messages
            WHERE session_id = ?
            """, (session_id,))

            return cursor.fetchone()[0]

    def get_total_session_cost(
        self,
        session_id
    ):

        with self._get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute("""
            SELECT SUM(total_cost)
            FROM usage_metrics
            WHERE session_id = ?
            """, (session_id,))

            result = cursor.fetchone()[0]

            return result if result else 0.0