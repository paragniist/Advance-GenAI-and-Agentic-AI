from fastmcp import FastMCP
import os
import sqlite3
from typing import Optional
from pydantic import BaseModel

# -----------------------
# PATHS
# -----------------------
BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "expenses.db")
CATEGORIES_PATH = os.path.join(BASE_DIR, "categories.json")

# -----------------------
# MCP SERVER
# -----------------------
mcp = FastMCP("ExpenseTracker")


# -----------------------
# DB INITIALIZATION
# -----------------------
def init_db() -> None:
    """Initialize the SQLite database."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS expenses(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                amount REAL NOT NULL,
                category TEXT NOT NULL,
                subcategory TEXT DEFAULT '',
                note TEXT DEFAULT ''
            )
            """
        )


# ✅ IMPORTANT: Run this at import time (fixes your issue)
init_db()


# -----------------------
# TOOLS
# -----------------------

# ✅ Pydantic Schemas
class AddExpenseInput(BaseModel):
    date: str
    amount: float
    category: str
    subcategory: Optional[str] = ""
    note: Optional[str] = ""

class DateRangeInput(BaseModel):
    start_date: str
    end_date: str

class SummaryInput(BaseModel):
    start_date: str
    end_date: str
    category: Optional[str] = None

class UpdateExpenseInput(BaseModel):
    id: int
    date: Optional[str] = None
    amount: Optional[float] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    note: Optional[str] = None    

class DeleteExpenseInput(BaseModel):
    id: int

@mcp.tool()
def add_expense(data: AddExpenseInput):
    '''Add a new expense entry to the database.'''
    with sqlite3.connect(DB_PATH) as c:
        cur = c.execute(
            "INSERT INTO expenses(date, amount, category, subcategory, note) VALUES (?,?,?,?,?)",
            (data.date, data.amount, data.category, data.subcategory, data.note)
        )
        return {"status": "ok", "id": cur.lastrowid}


@mcp.tool()
def list_expenses(data: DateRangeInput):
    '''List expense entries within an inclusive date range.'''
    with sqlite3.connect(DB_PATH) as c:
        cur = c.execute(
            """
            SELECT id, date, amount, category, subcategory, note
            FROM expenses
            WHERE date BETWEEN ? AND ?
            ORDER BY id ASC
            """,
            (data.start_date, data.end_date)
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


@mcp.tool()
def summarize(data: SummaryInput):
    '''Summarize expenses by category within an inclusive date range.'''
    with sqlite3.connect(DB_PATH) as c:
        query = """
            SELECT category, SUM(amount) AS total_amount
            FROM expenses
            WHERE date BETWEEN ? AND ?
        """
        params = [data.start_date, data.end_date]

        if data.category:
            query += " AND category = ?"
            params.append(data.category)

        query += " GROUP BY category ORDER BY category ASC"

        cur = c.execute(query, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    
@mcp.tool()
def update_expense(data: UpdateExpenseInput):
    '''
    Update an existing expense entry by ID.
    Only provided fields will be updated.
    '''

    fields = []
    values = []

    # Dynamically build query
    if data.date is not None:
        fields.append("date = ?")
        values.append(data.date)

    if data.amount is not None:
        fields.append("amount = ?")
        values.append(data.amount)

    if data.category is not None:
        fields.append("category = ?")
        values.append(data.category)

    if data.subcategory is not None:
        fields.append("subcategory = ?")
        values.append(data.subcategory)

    if data.note is not None:
        fields.append("note = ?")
        values.append(data.note)

    # ❌ Nothing to update
    if not fields:
        return {"status": "error", "message": "No fields provided to update"}

    values.append(data.id)

    query = f"""
        UPDATE expenses
        SET {', '.join(fields)}
        WHERE id = ?
    """

    with sqlite3.connect(DB_PATH) as c:
        cur = c.execute(query, values)

        if cur.rowcount == 0:
            return {"status": "error", "message": "Expense not found"}

        return {"status": "success", "updated_id": data.id}   

@mcp.tool()
def delete_expense(data: DeleteExpenseInput):
    '''
    Delete an expense entry by ID.
    '''

    with sqlite3.connect(DB_PATH) as c:
        cur = c.execute(
            "DELETE FROM expenses WHERE id = ?",
            (data.id,)
        )

        if cur.rowcount == 0:
            return {
                "status": "error",
                "message": "Expense not found"
            }

        return {
            "status": "success",
            "deleted_id": data.id
        }     


# -----------------------
# RESOURCE
# -----------------------

@mcp.resource("expense://categories", mime_type="application/json")
def categories() -> str:
    """
    Returns available expense categories.
    Reads categories.json dynamically.
    """
    with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
        return f.read()


# -----------------------
# PROMPT
# -----------------------

@mcp.prompt("expense-assistant")
def expense_assistant() -> dict:
    """
    Base system prompt for the expense assistant.
    """
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful and intelligent assistant.\n\n"

                    "You can manage expenses using available tools such as adding, listing, "
                    "and summarizing expenses.\n\n"

                    "Guidelines:\n"
                    "- Use tools ONLY when the query is related to expense management.\n"
                    "- For general or unrelated questions, answer directly using your knowledge.\n"
                    "- Do NOT refuse general questions.\n"
                    "- If a tool is not required, respond normally.\n"
                    "- When using tools, ensure correct arguments and formats.\n"
                    "- Dates must follow the YYYY-MM-DD format.\n\n"

                    "Your goal is to be helpful, flexible, and accurate."
                )
            }
        ]
    }


# -----------------------
# MAIN (only for direct run)
# -----------------------

if __name__ == "__main__":
    mcp.run()