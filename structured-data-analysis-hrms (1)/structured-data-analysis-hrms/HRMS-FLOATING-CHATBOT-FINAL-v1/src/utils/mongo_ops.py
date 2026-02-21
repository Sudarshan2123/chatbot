import pymongo
from typing import Any,Dict,List,Optional
import pymongo.errors
from src.entity import Base_Config
from src.config.configuration import ConfigurationManager
config_manager = ConfigurationManager()
base_config = config_manager.get_base_config()

def execute_query(
        query:str,
        params:Optional[Dict[str,Any]]=None,
        db_name:str="HR_bot",
        collection_name:str="User",
        fetch:str="none",
    )->Optional[list[Dict[str,Any]]]:
    client=None
    try:
        client = pymongo.MongoClient(base_config.MONGODB_URI)
        db=client[db_name]
        collection=db[collection_name]

        if query=="find_one":
             if params:
                  return collection.find_one(params)
             else:
                  return collection.find_one()
        elif query=="find":
             if params:
                  results=collection.find(params)
             else:
                  results=collection.find()
             if fetch=="all":
                  return list(results)
        elif query == "insert_one":
            if params:
                return collection.insert_one(params)
        elif query == "insert_many":
            if isinstance(params, list):
                return collection.insert_many(params)
        elif query == "update_one":
            if isinstance(params, dict) and "filter" in params and "update" in params:
                return collection.update_one(params["filter"], {"$set": params["update"]})
        elif query == "update_many":
            if isinstance(params, dict) and "filter" in params and "update" in params:
                return collection.update_many(params["filter"], {"$set": params["update"]})
        elif query == "delete_one":
            if params:
                return collection.delete_one(params)
        elif query == "delete_many":
            if params:
                return collection.delete_many(params)
        else:
            print(f"Unsupported MongoDB query: {query}")
            return None
    
    except pymongo.errors.ConnectionFailure as e:
        print(f"could not able to connect to mongo db:{e}")
    except pymongo.errors.OperationFailure as e:
        print(f"Mongo Operation Failed:{e}")
    finally:
        if client:
            client.close()

    return None

def dynamic_operation(operation_type: str, db_name: str, collection_name: str,
                      filter_conditions: Optional[Dict[str, Any]] = None,
                      update_values: Optional[Dict[str, Any]] = None,
                      insert_values: Optional[Dict[str, Any]] = None,
                      delete_condition: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    """
    A function to perform SELECT, INSERT, UPDATE, or DELETE dynamically in MongoDB based on parameters.

    :param operation_type: The operation type ('SELECT', 'INSERT', 'UPDATE', 'DELETE')
    :param db_name: The name of the MongoDB database.
    :param collection_name: The name of the MongoDB collection.
    :param filter_conditions: Conditions for SELECT, UPDATE, DELETE (e.g., {'age': 25})
    :param update_values: A dictionary of values to update (e.g., {'name': 'John', 'age': 30})
    :param insert_values: Values for a new document to insert (e.g., {'name': 'Alice', 'age': 22})
    :param delete_condition: Condition to delete (e.g., {'_id': ObjectId('some_id')})
    :return: The result of the operation or None
    """
    if operation_type == 'SELECT':
        # SELECT operation, returns filtered documents
        return execute_query(
            query='find',
            params=filter_conditions,
            db_name=db_name,
            collection_name=collection_name,
            fetch='all'
        )
    elif operation_type == 'INSERT':
        # INSERT operation, inserts a new document
        if insert_values:
            return execute_query(
                query='insert_one',
                params=insert_values,
                db_name=db_name,
                collection_name=collection_name
            )
        return None
    elif operation_type == 'UPDATE':
        # UPDATE operation, updates existing documents
        if filter_conditions and update_values:
            return execute_query(
                query='update_many',
                params={'filter': filter_conditions, 'update': update_values},
                db_name=db_name,
                collection_name=collection_name
            )
        return None
    elif operation_type == 'DELETE':
        # DELETE operation, deletes documents
        if delete_condition:
            return execute_query(
                query='delete_many',
                params=delete_condition,
                db_name=db_name,
                collection_name=collection_name
            )
        return None
    return None

# Example Usage (assuming MongoDB is running locally)

# # Initialize database and collection names
#DB_NAME = self.config.Mo
#COLLECTION_NAME = "users"

# # 1. SELECT operation (Get users where age = 25)
#selected_users = dynamic_operation('SELECT', DB_NAME, COLLECTION_NAME, filter_conditions={'age': 25})
#print("Selected Users:", selected_users)

# # 2. INSERT operation (Add a new user)
#new_user = dynamic_operation('INSERT', DB_NAME, COLLECTION_NAME, insert_values={'name': 'Alice', 'age': 22})
#print("Insert Result:", new_user.inserted_id if new_user else None)

# # 3. UPDATE operation (Update users with age = 25 to age = 30)
#updated_users = dynamic_operation('UPDATE', DB_NAME, COLLECTION_NAME, filter_conditions={'age': 25}, update_values={'age': 30})
#print("Update Result:", updated_users.modified_count if updated_users else None)

# # 4. DELETE operation (Delete user named 'Alice')
#deleted_users = dynamic_operation('DELETE', DB_NAME, COLLECTION_NAME, delete_condition={'name': 'Alice'})
#print("Delete Result:", deleted_users.deleted_count if deleted_users else None)