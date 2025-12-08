import sqlite3
from typing import Any, Tuple, List, Optional
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

def execute_query(
    query: str,
    params: Optional[Tuple[Any, ...]] = None,
    db_path: str = "database.db",
    fetch: str = "none",
) -> Optional[List[Tuple[Any]]]:
    """
    General function to interact with SQLite.

    Args:
        query (str): The SQL query to execute.
        params (Optional[Tuple[Any, ...]]): Parameters to use in the query.
        db_path (str): Path to the SQLite database file.
        fetch (str): Determines whether to fetch results. Options: 'none', 'one', 'all'.
                     - 'none': Executes query without fetching results.
                     - 'one': Fetches a single row.
                     - 'all': Fetches all rows.

    Returns:
        Optional[List[Tuple[Any]]]: Fetched data if `fetch` is 'one' or 'all', otherwise None.
    """
    connection = None
    cursor = None
    try:
        # Connect to the database
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        if query.strip().lower().startswith(("insert", "update", "delete")):
            connection.commit()
        if fetch == "one":
            return cursor.fetchone()
        elif fetch == "all":
            return cursor.fetchall()

    except sqlite3.OperationalError as e:
        print(f"OperationalError: {e}")
    except sqlite3.DatabaseError as e:
        print(f"DatabaseError: {e}")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

    return None




def create_session(DB_connetion):
    Base = declarative_base()
    engine = create_engine(DB_connetion, echo=True)  # Using SQLite file example.db
    Session = sessionmaker(bind=engine)
    session = Session()
    Base.metadata.create_all(engine)
    return session

# Create the table in the database if it doesn't exist


def dynamic_operation(operation_type,DB_connetion, model, filter_conditions=None, update_values=None, insert_values=None, delete_condition=None):
    """
    A function to perform SELECT, INSERT, UPDATE, or DELETE dynamically based on parameters.

    :param operation_type: The operation type ('SELECT', 'INSERT', 'UPDATE', 'DELETE')
    :param model: The SQLAlchemy model to interact with (e.g., User)
    :param filter_conditions: Conditions for SELECT, UPDATE, DELETE (e.g., {'id': 1})
    :param update_values: A dictionary of values to update (e.g., {'name': 'John', 'age': 25})
    :param insert_values: Values for a new record to insert (e.g., {'name': 'Jane', 'age': 28})
    :param delete_condition: Condition to delete (e.g., {'id': 1})
    :return: The result of the operation or None
    """
    session=create_session(DB_connetion)
    if operation_type == 'SELECT':
        # SELECT operation, returns filtered records
        query = session.query(model)
        if filter_conditions:
            for key, value in filter_conditions.items():
                query = query.filter(getattr(model, key) == value)
        return query.all()  # Return all matching records
    
    elif operation_type == 'INSERT':
        # INSERT operation, inserts a new record
        if insert_values:
            new_record = model(**insert_values)
            session.add(new_record)
            session.commit()
            return new_record
        return None
    
    elif operation_type == 'UPDATE':
        # UPDATE operation, updates existing records
        if filter_conditions and update_values:
            query = session.query(model)
            for key, value in filter_conditions.items():
                query = query.filter(getattr(model, key) == value)
            records_to_update = query.all()
            
            for record in records_to_update:
                for key, value in update_values.items():
                    setattr(record, key, value)
            session.commit()
            return records_to_update
        return None
    
    elif operation_type == 'DELETE':
        # DELETE operation, deletes records
        if delete_condition:
            query = session.query(model)
            for key, value in delete_condition.items():
                query = query.filter(getattr(model, key) == value)
            records_to_delete = query.all()
            
            for record in records_to_delete:
                session.delete(record)
            session.commit()
            return records_to_delete
        return None
    session.close()



