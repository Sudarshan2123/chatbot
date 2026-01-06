"""
Combined Data Preparation Script
Combines functionality from:
- 1_los_data_prep.ipynb
- 2_lms_data_prep.ipynb  
- 3_hybrid_data_prep_optimized.ipynb
"""

import pandas as pd
import numpy as np
import os
import sqlalchemy
import joblib
import random
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Embedding, Flatten
from tensorflow.keras.optimizers import Adam

# Constants
TWO_CLASS_STATUS_MAP = {0: 'Paid', 1: 'Not Paid'}
GRACE_PERIOD_DAYS = 2
MAX_SEQUENCE_LENGTH = 8
TEST_RATIO = 0.20
FIT_SAMPLE_RATIO = 0.5

class DataPreparation:
    def __init__(self):
        self.df_los = None
        self.df_lms = None
        self.df_combined = None
        self.preprocessor = None
        
    def load_los_data(self):
        """Load and preprocess LOS data from PostgreSQL"""
        print("--- Loading LOS Data from PostgreSQL ---")
        
        # Database connection
        db_username = 'ml_db'
        db_password = 'pass%401234'
        db_host = '10.192.5.43'
        db_port = '5432'
        db_name = 'postgres'
        
        conn_string = f"postgresql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"
        engine = sqlalchemy.create_engine(conn_string)
        
        try:
            query = """
            SELECT *
            FROM "Ashirvad"
            WHERE "LOAN_DATE" BETWEEN '2023-01-01' AND '2023-12-31';
            """
            self.df_los = pd.read_sql_query(query, con=engine)
            print(f"LOS Data loaded. Total shape: {self.df_los.shape}")
        except Exception as e:
            print(f"Database connection or query failed: {e}")
            self.df_los = pd.DataFrame()
            
    def preprocess_los_data(self):
        """Clean and preprocess LOS data"""
        if self.df_los.empty:
            return
            
        print("--- Preprocessing LOS Data ---")
        
        # Drop unnecessary columns
        columns_to_drop = [
            'CUSTOMER_ID', 'CUSTOMER_NAME', 'BRANCH_ID', 'TEMP_CUST_ID', 'PHONE1', 'PHONE2',
            'HOUSE_NAME', 'LOCALITY', 'STREET', 'ALT_HOUSE_NAME', 'ALT_LOCALITY', 'ALT_STREET',
            'CENTER_ID', 'ALT_PIN_CODE', 'MARITAL_STATUS', 'LOAN_STATUS', 'LOAN_STATUS_DESC',
            'CLS_DT', 'CIBIL_ID', 'NPA_FLAG', 'NPA_FROM_DATE', 'NPA_TO_DATE', 'NPA_STATUS',
            'OCCUPATION_ID', 'APPLICATION_ID', 'TENURE_in_months', 'emi_paid', 'loan_paid_percentage',
            'NPA_STATUS_UPDATED', 'NPA_STATUS_UPDATED_1', 'NPA_STATUS_UPDATED_2', 'YEAR',
            'CIBIL_SCORE', 'CUSTOMER_GRADING_SCORE', 'BRANCH_NAME', 'CENTER_NAME'
        ]
        
        self.df_los.drop(columns=[col for col in columns_to_drop if col in self.df_los.columns], 
                        inplace=True, errors='ignore')
        
        # Convert PIN_CODE to string
        if 'PIN_CODE' in self.df_los.columns:
            self.df_los['PIN_CODE'] = self.df_los['PIN_CODE'].astype(str)
        
        # Handle date columns
        self.df_los['DATE_OF_BIRTH'] = pd.to_datetime(self.df_los['DATE_OF_BIRTH'], errors='coerce')
        self.df_los['LOAN_DATE'] = pd.to_datetime(self.df_los['LOAN_DATE'], errors='coerce')
        
        # Calculate age
        self.df_los['AGE'] = (self.df_los['LOAN_DATE'] - self.df_los['DATE_OF_BIRTH']).dt.days // 365.25
        self.df_los.drop(columns=['DATE_OF_BIRTH'], inplace=True)
        
        # Filter out inactive customers
        self.df_los = self.df_los[self.df_los['CUSTOMER_FLAG'] != 'X'].copy()
        
        # Handle missing values
        numerical_cols = ['TOTAL_EXPENSE', 'TOTAL_INCOME', 'AGE']
        for col in numerical_cols:
            if col in self.df_los.columns and self.df_los[col].isnull().sum() > 0:
                median_val = self.df_los[col].median()
                self.df_los[col] = self.df_los[col].fillna(median_val)
        
        if 'OCCUPATION_NAME' in self.df_los.columns:
            self.df_los['OCCUPATION_NAME'] = self.df_los['OCCUPATION_NAME'].fillna('UNKNOWN')
        
        # Consolidate marital status
        if 'MARITAL_STATUS_NAME' in self.df_los.columns:
            self.df_los['MARITAL_STATUS_NAME'] = self.df_los['MARITAL_STATUS_NAME'].replace(
                ['UNMARRIED', 'SINGLE'], 'UNMARRIED/SINGLE'
            )
        
        print(f"LOS data preprocessed. Final shape: {self.df_los.shape}")
        
    def load_lms_data(self):
        """Load and preprocess LMS data from Excel files"""
        print("--- Loading LMS Data from Excel Files ---")
        
        folder_path = "CorrectedCollection2023formatted"
        excel_files = {
            'jan': [f'{folder_path}/jan1_2023.xlsx', f'{folder_path}/jan2_2023.xlsx'],
            'feb': [f'{folder_path}/feb1_2023.xlsx', f'{folder_path}/feb2_2023.xlsx'],
            'mar': [f'{folder_path}/mar1_2023.xlsx', f'{folder_path}/mar2_2023.xlsx', f'{folder_path}/mar3_2023.xlsx'],
            'apr': [f'{folder_path}/apr1_2023.xlsx', f'{folder_path}/apr2_2023.xlsx', f'{folder_path}/apr3_2023.xlsx'],
            'may': [f'{folder_path}/may1_2023.xlsx'],
            'jun': [f'{folder_path}/jun1_2023.xlsx'],
            'jul': [f'{folder_path}/jul1_2023.xlsx', f'{folder_path}/jul2_2023.xlsx'],
            'aug': [f'{folder_path}/aug1_2023.xlsx', f'{folder_path}/aug2_2023.xlsx', f'{folder_path}/aug3_2023.xlsx']
        }
        
        all_dfs = []
        for month, files in excel_files.items():
            print(f"Loading files for {month.capitalize()}...")
            for file_name in files:
                if os.path.exists(file_name):
                    try:
                        df_month = pd.read_excel(file_name)
                        all_dfs.append(df_month)
                        print(f"✅ Successfully loaded: {file_name}")
                    except Exception as e:
                        print(f"❌ Error loading {file_name}: {e}")
                else:
                    print(f"⚠️ Warning: {file_name} not found.")
        
        if all_dfs:
            self.df_lms = pd.concat(all_dfs, ignore_index=True)
            print(f"LMS Data loaded. Total shape: {self.df_lms.shape}")
        else:
            self.df_lms = pd.DataFrame()
            
    def preprocess_lms_data(self):
        """Feature engineering for LMS data"""
        if self.df_lms.empty:
            return
            
        print("--- Feature Engineering LMS Data ---")
        
        # Convert date columns
        self.df_lms['DUE_DATE'] = pd.to_datetime(self.df_lms['DUE_DATE'])
        self.df_lms['LOAN_DATE'] = pd.to_datetime(self.df_lms['LOAN_DATE'])
        
        # Create payment status
        self.df_lms['IS_UNPAID'] = (self.df_lms['STATUS'] == 1).astype(int)
        
        # Calculate days late
        self.df_lms['DAYS_LATE'] = 0  # Simplified for this example
        
        # Calculate paid ratio
        self.df_lms['PAID_RATIO'] = np.where(
            self.df_lms['INSTALLMENT_AMOUNT'] > 0,
            self.df_lms['PAID_AMOUNT'] / self.df_lms['INSTALLMENT_AMOUNT'],
            1.0
        )
        
        # Create payment score
        self.df_lms['PAYMENT_SCORE'] = np.where(
            self.df_lms['IS_UNPAID'] == 1, -100,
            np.where(self.df_lms['PAID_RATIO'] >= 1.0, 1.5, self.df_lms['PAID_RATIO'] * 1.5)
        )
        
        # Create behavior labels
        self.df_lms['CURRENT_EMI_BEHAVIOR_LABEL'] = (self.df_lms['IS_UNPAID'] == 1).astype(int)
        
        # Create next EMI label (target)
        self.df_lms = self.df_lms.sort_values(['LOAN_ID', 'INSTALLMENT_NO'])
        self.df_lms['NEXT_EMI_LABEL'] = self.df_lms.groupby('LOAN_ID')['CURRENT_EMI_BEHAVIOR_LABEL'].shift(-1)
        self.df_lms['NEXT_EMI_LABEL'] = self.df_lms['NEXT_EMI_LABEL'].fillna(0).astype(int)
        
        # Additional features
        self.df_lms['DELTA_DAYS_LATE'] = 0  # Simplified
        self.df_lms['COMPOSITE_RISK'] = self.df_lms['PAYMENT_SCORE'] * -1
        self.df_lms['PAYMENT_SCORE_RANK'] = 1  # Simplified
        
        # Calculate days between due dates
        self.df_lms['DAYS_BETWEEN_DUE_DATES'] = self.df_lms.groupby('LOAN_ID')['DUE_DATE'].diff().dt.days
        self.df_lms['DAYS_BETWEEN_DUE_DATES'].fillna(0, inplace=True)
        
        # Create repayment schedule category
        self.df_lms['REPAYMENT_SCHEDULE_CAT'] = 'Monthly'  # Simplified
        
        # Create loan schedule type
        self.df_lms['LOAN_SCHEDULE_TYPE'] = 'Monthly'  # Simplified
        
        print(f"LMS data feature engineering complete. Shape: {self.df_lms.shape}")
        
    def combine_datasets(self):
        """Combine LOS and LMS datasets"""
        print("--- Combining LOS and LMS Data ---")
        
        if self.df_los.empty or self.df_lms.empty:
            print("❌ Cannot combine datasets - one or both are empty")
            return
            
        # Rename columns to avoid conflicts
        rename_map_los = {
            'LOAN_AMOUNT': 'LOAN_AMOUNT_STATIC',
            'TENURE': 'TENURE_STATIC',
            'INTEREST_RATE': 'INTEREST_RATE_STATIC'
        }
        self.df_los.rename(columns=rename_map_los, inplace=True)
        
        # Merge datasets
        self.df_combined = pd.merge(self.df_los, self.df_lms, on='LOAN_ID', how='inner')
        print(f"✅ Combined datasets. Shape: {self.df_combined.shape}")
        
        # Filter out later months (keep only Jan-Aug)
        self.df_combined['LOAN_MONTH'] = self.df_combined['LOAN_DATE_x'].dt.month
        months_to_keep = [1, 2, 3, 4, 5, 6, 7, 8]
        self.df_combined = self.df_combined[self.df_combined['LOAN_MONTH'].isin(months_to_keep)].copy()
        del self.df_combined['LOAN_MONTH']
        
        # Clean up column names
        if 'LOAN_DATE_y' in self.df_combined.columns:
            del self.df_combined['LOAN_DATE_y']
        self.df_combined.rename(columns={"LOAN_DATE_x": "LOAN_DATE"}, inplace=True)
        
        print(f"Final combined dataset shape: {self.df_combined.shape}")
        
    def prepare_features(self):
        """Prepare features for modeling"""
        print("--- Preparing Features for Modeling ---")
        
        if self.df_combined is None or self.df_combined.empty:
            print("❌ No combined dataset available")
            return
            
        # Create rolling features
        ROLLING_WINDOW_SIZE = 3
        self.df_combined['RECENT_PAYMENT_SCORE'] = self.df_combined.groupby('LOAN_ID')['PAYMENT_SCORE'].rolling(
            window=ROLLING_WINDOW_SIZE, min_periods=1
        ).mean().reset_index(level=0, drop=True).astype(np.float32)
        
        self.df_combined['RECENT_PAYMENT_SCORE'] = self.df_combined.groupby('LOAN_ID')['RECENT_PAYMENT_SCORE'].shift(1)
        overall_mean_score = self.df_combined['PAYMENT_SCORE'].mean()
        self.df_combined['RECENT_PAYMENT_SCORE'].fillna(overall_mean_score, inplace=True)
        
        # Encode customer flag
        FLAG_ORDER = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
        self.df_combined['CUSTOMER_FLAG_ENCODED'] = self.df_combined['CUSTOMER_FLAG'].map(FLAG_ORDER).astype(np.int8)
        del self.df_combined['CUSTOMER_FLAG']
        
        # Label encoding for embeddings
        self.df_combined['OCCUPATION_NAME'] = self.df_combined['OCCUPATION_NAME'].astype('category')
        self.df_combined['LOAN_PURPOSE'] = self.df_combined['LOAN_PURPOSE'].astype('category')
        
        # Save category mappings
        occupation_categories = self.df_combined['OCCUPATION_NAME'].cat.categories
        purpose_categories = self.df_combined['LOAN_PURPOSE'].cat.categories
        
        occupation_mapping = dict(enumerate(occupation_categories))
        purpose_mapping = dict(enumerate(purpose_categories))
        
        category_mappings = {'occupation': occupation_mapping, 'purpose': purpose_mapping}
        joblib.dump(category_mappings, 'embedding_category_mappings.pkl')
        
        # Create encoded columns
        self.df_combined['OCCUPATION_NAME_ENCODED'] = self.df_combined['OCCUPATION_NAME'].cat.codes + 1
        self.df_combined['LOAN_PURPOSE_ENCODED'] = self.df_combined['LOAN_PURPOSE'].cat.codes + 1
        self.df_combined['OCCUPATION_NAME_ENCODED'] = self.df_combined['OCCUPATION_NAME_ENCODED'].astype(np.int16)
        self.df_combined['LOAN_PURPOSE_ENCODED'] = self.df_combined['LOAN_PURPOSE_ENCODED'].astype(np.int16)
        
        # Drop original columns
        del self.df_combined['OCCUPATION_NAME']
        del self.df_combined['LOAN_PURPOSE']
        
        # One-hot encode repayment schedule
        self.df_combined = pd.get_dummies(self.df_combined, columns=['REPAYMENT_SCHEDULE_CAT'], prefix='REPAYMENT_CAT')
        
        print("✅ Feature preparation complete")
        
    def split_data(self):
        """Split data by LOAN_ID"""
        print(f"--- Splitting Data ({int(TEST_RATIO*100)}% for Test) ---")
        
        all_loan_ids = self.df_combined['LOAN_ID'].unique()
        random.seed(42)
        random.shuffle(all_loan_ids)
        split_point = int(len(all_loan_ids) * (1 - TEST_RATIO))
        train_ids = all_loan_ids[:split_point]
        test_ids = all_loan_ids[split_point:]
        
        self.train_df = self.df_combined[self.df_combined['LOAN_ID'].isin(train_ids)].copy()
        self.test_df = self.df_combined[self.df_combined['LOAN_ID'].isin(test_ids)].copy()
        
        # Remove date columns
        for col in ['LOAN_DATE', 'DUE_DATE']:
            if col in self.train_df.columns:
                del self.train_df[col]
            if col in self.test_df.columns:
                del self.test_df[col]
        
        print(f"Train Records: {self.train_df.shape} | Test Records: {self.test_df.shape}")
        
    def apply_preprocessing(self):
        """Apply scaling and encoding"""
        print("--- Applying Preprocessing ---")
        
        # Define feature groups
        SEQUENTIAL_COLS_NUMERICAL = [
            'INSTALLMENT_NO', 'INSTALLMENT_AMOUNT', 'DAYS_LATE', 'DAYS_BETWEEN_DUE_DATES',
            'PAID_RATIO', 'DELTA_DAYS_LATE', 'PAYMENT_SCORE', 'COMPOSITE_RISK',
            'RECENT_PAYMENT_SCORE', 'PAYMENT_SCORE_RANK', 'IS_UNPAID', 'CURRENT_EMI_BEHAVIOR_LABEL'
        ]
        
        STATIC_COLS_NUMERICAL = [
            'TOTAL_INCOME', 'TOTAL_EXPENSE', 'LOAN_AMOUNT_STATIC', 'AGE', 'CYCLE', 'CUSTOMER_FLAG_ENCODED'
        ]
        
        REPAYMENT_CAT_OHE_COLS = [col for col in self.train_df.columns if col.startswith('REPAYMENT_CAT_')]
        NUMERICAL_FEATURES_FINAL = SEQUENTIAL_COLS_NUMERICAL + STATIC_COLS_NUMERICAL + REPAYMENT_CAT_OHE_COLS
        
        STATIC_COLS_OHE = ['MARITAL_STATUS_NAME', 'STATE_NAME', 'LOAN_SCHEDULE_TYPE']
        TARGET_COL = 'NEXT_EMI_LABEL'
        STATIC_COLS_EMBEDDING_FINAL = ['OCCUPATION_NAME_ENCODED', 'LOAN_PURPOSE_ENCODED']
        
        PASSTHROUGH_COLS = ['LOAN_ID', TARGET_COL] + STATIC_COLS_EMBEDDING_FINAL
        
        # Create preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), NUMERICAL_FEATURES_FINAL),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), STATIC_COLS_OHE)
            ],
            remainder='passthrough'
        )
        
        # Fit on sample
        sample_indices = np.random.choice(self.train_df.index, size=int(len(self.train_df) * FIT_SAMPLE_RATIO), replace=False)
        train_df_sample = self.train_df.loc[sample_indices].copy()
        
        self.preprocessor.fit(train_df_sample)
        joblib.dump(self.preprocessor, 'preprocessor.pkl')
        
        # Transform data
        train_transformed = self.preprocessor.transform(self.train_df)
        test_transformed = self.preprocessor.transform(self.test_df)
        
        # Get feature names
        def get_feature_names(preprocessor, numerical_cols, static_ohe_cols, passthrough_cols):
            num_features = numerical_cols
            ohe_transformer = preprocessor.named_transformers_['cat']
            ohe_feature_names = list(ohe_transformer.get_feature_names_out(static_ohe_cols))
            passthrough_features = passthrough_cols
            all_features = num_features + ohe_feature_names + passthrough_features
            return all_features
        
        ALL_FINAL_COLS = get_feature_names(
            self.preprocessor, NUMERICAL_FEATURES_FINAL, STATIC_COLS_OHE, PASSTHROUGH_COLS
        )
        
        # Convert to DataFrames
        self.X_train_df = pd.DataFrame(train_transformed, columns=ALL_FINAL_COLS, index=self.train_df.index)
        self.X_test_df = pd.DataFrame(test_transformed, columns=ALL_FINAL_COLS, index=self.test_df.index)
        
        # Extract targets
        self.y_train = self.X_train_df[TARGET_COL].values.astype(np.int8)
        self.y_test = self.X_test_df[TARGET_COL].values.astype(np.int8)
        self.X_train_df.drop(columns=[TARGET_COL], inplace=True)
        self.X_test_df.drop(columns=[TARGET_COL], inplace=True)
        
        print("✅ Preprocessing complete")
        
    def reshape_for_lstm(self):
        """Reshape data for LSTM model"""
        print("--- Reshaping Data for LSTM ---")
        
        # Define feature groups for reshaping
        LSTM_INPUT_COLS = [
            'INSTALLMENT_NO', 'INSTALLMENT_AMOUNT', 'DAYS_LATE', 'DAYS_BETWEEN_DUE_DATES',
            'PAID_RATIO', 'DELTA_DAYS_LATE', 'PAYMENT_SCORE', 'COMPOSITE_RISK',
            'RECENT_PAYMENT_SCORE', 'PAYMENT_SCORE_RANK', 'IS_UNPAID', 'CURRENT_EMI_BEHAVIOR_LABEL'
        ] + [col for col in self.X_train_df.columns if col.startswith('REPAYMENT_CAT_')]
        
        STATIC_DENSE_COLS = [
            'TOTAL_INCOME', 'TOTAL_EXPENSE', 'LOAN_AMOUNT_STATIC', 'AGE', 'CYCLE', 'CUSTOMER_FLAG_ENCODED'
        ] + [col for col in self.X_train_df.columns if any(col.startswith(base) for base in ['MARITAL_STATUS_NAME_', 'STATE_NAME_', 'LOAN_SCHEDULE_TYPE_'])]
        
        STATIC_EMBEDDING_COLS = ['OCCUPATION_NAME_ENCODED', 'LOAN_PURPOSE_ENCODED']
        
        def reshape_and_pad_final(X_df, y_array, lstm_cols, static_dense_cols, embedding_cols, max_len):
            grouped = X_df.groupby('LOAN_ID')
            loan_ids = list(grouped.groups.keys())
            
            X_lstm = np.zeros((len(loan_ids), max_len, len(lstm_cols)), dtype=np.float32)
            X_static_dense = np.zeros((len(loan_ids), len(static_dense_cols)), dtype=np.float32)
            X_static_embed = np.zeros((len(loan_ids), len(embedding_cols)), dtype=np.int16)
            y_final = np.zeros(len(loan_ids), dtype=np.int8)
            
            for i, loan_id in enumerate(loan_ids):
                loan_data = grouped.get_group(loan_id)
                
                # LSTM data
                sequence = loan_data[lstm_cols].values
                if len(sequence) >= max_len:
                    X_lstm[i, :, :] = sequence[-max_len:]
                else:
                    X_lstm[i, -len(sequence):, :] = sequence
                
                # Static data
                last_record = loan_data.iloc[-1]
                X_static_dense[i, :] = last_record[static_dense_cols].values
                X_static_embed[i, :] = last_record[embedding_cols].values.astype(np.int16)
                
                # Target
                last_index = loan_data.index[-1]
                y_final[i] = y_array[np.where(X_df.index == last_index)[0][0]]
            
            return X_lstm, X_static_dense, X_static_embed, y_final, loan_ids
        
        # Reshape training data
        self.X_train_lstm, self.X_train_static_dense, self.X_train_static_embed, self.y_train_final, self.train_loan_ids = reshape_and_pad_final(
            self.X_train_df, self.y_train, LSTM_INPUT_COLS, STATIC_DENSE_COLS, STATIC_EMBEDDING_COLS, MAX_SEQUENCE_LENGTH
        )
        
        # Reshape test data
        self.X_test_lstm, self.X_test_static_dense, self.X_test_static_embed, self.y_test_final, self.test_loan_ids = reshape_and_pad_final(
            self.X_test_df, self.y_test, LSTM_INPUT_COLS, STATIC_DENSE_COLS, STATIC_EMBEDDING_COLS, MAX_SEQUENCE_LENGTH
        )
        
        print(f"Training Set Shapes:")
        print(f"  LSTM Input: {self.X_train_lstm.shape}")
        print(f"  Dense Input: {self.X_train_static_dense.shape}")
        print(f"  Embedding Input: {self.X_train_static_embed.shape}")
        print(f"  Target: {self.y_train_final.shape}")
        
    def build_model(self):
        """Build hybrid LSTM model"""
        print("--- Building Hybrid Model ---")
        
        # Input dimensions
        LSTM_FEATURES = self.X_train_lstm.shape[2]
        STATIC_DENSE_FEATURES = self.X_train_static_dense.shape[1]
        
        # Vocabulary sizes for embeddings
        VOCAB_SIZE_OCCUPATION = int(self.X_train_static_embed[:, 0].max() + 1)
        VOCAB_SIZE_PURPOSE = int(self.X_train_static_embed[:, 1].max() + 1)
        EMBEDDING_DIM = 8
        
        # LSTM Branch
        lstm_input = Input(shape=(MAX_SEQUENCE_LENGTH, LSTM_FEATURES), name='lstm_input')
        lstm_output = LSTM(units=32, activation='relu', name='lstm_layer')(lstm_input)
        lstm_output_dense = Dense(units=16, activation='relu', name='lstm_dense_output')(lstm_output)
        
        # Static Dense Branch
        static_dense_input = Input(shape=(STATIC_DENSE_FEATURES,), name='static_dense_input')
        static_dense_output = Dense(units=16, activation='relu', name='static_dense_layer')(static_dense_input)
        
        # Embedding Branch
        embed_input = Input(shape=(len(['OCCUPATION_NAME_ENCODED', 'LOAN_PURPOSE_ENCODED']),), name='embedding_input')
        
        occupation_input = tf.slice(embed_input, [0, 0], [-1, 1])
        purpose_input = tf.slice(embed_input, [0, 1], [-1, 1])
        
        embed_occupation = Embedding(input_dim=VOCAB_SIZE_OCCUPATION, output_dim=EMBEDDING_DIM, name='occupation_embedding')(occupation_input)
        embed_purpose = Embedding(input_dim=VOCAB_SIZE_PURPOSE, output_dim=EMBEDDING_DIM, name='purpose_embedding')(purpose_input)
        
        embed_flat_occupation = Flatten()(embed_occupation)
        embed_flat_purpose = Flatten()(embed_purpose)
        embed_concat = Concatenate(name='embed_concat_output')([embed_flat_occupation, embed_flat_purpose])
        
        # Combine static branches
        static_combined = Concatenate(name='static_combined_output')([static_dense_output, embed_concat])
        static_combined_final = Dense(units=16, activation='relu', name='static_final_dense')(static_combined)
        
        # Final layers
        merged = Concatenate(name='merged_output')([lstm_output_dense, static_combined_final])
        final_dense_1 = Dense(units=32, activation='relu', name='final_dense_1')(merged)
        final_dense_2 = Dense(units=16, activation='relu', name='final_dense_2')(final_dense_1)
        output_layer = Dense(units=1, activation='sigmoid', name='final_prediction')(final_dense_2)
        
        # Create and compile model
        self.model = Model(inputs=[lstm_input, static_dense_input, embed_input], outputs=output_layer)
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        print("✅ Model built and compiled")
        self.model.summary()
        
    def train_model(self, epochs=10, batch_size=512):
        """Train the model"""
        print("--- Training Model ---")
        
        # Convert labels to float32
        y_train_final = self.y_train_final.astype(np.float32)
        y_test_final = self.y_test_final.astype(np.float32)
        
        # Define inputs
        train_inputs = {
            'lstm_input': self.X_train_lstm,
            'static_dense_input': self.X_train_static_dense,
            'embedding_input': self.X_train_static_embed
        }
        
        test_inputs = {
            'lstm_input': self.X_test_lstm,
            'static_dense_input': self.X_test_static_dense,
            'embedding_input': self.X_test_static_embed
        }
        
        # Train model
        history = self.model.fit(
            train_inputs,
            y_train_final,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            verbose=1
        )
        
        # Evaluate
        loss, accuracy, auc = self.model.evaluate(test_inputs, y_test_final, batch_size=batch_size, verbose=0)
        print(f"✅ Test Loss: {loss:.4f}")
        print(f"✅ Test Accuracy: {accuracy:.4f}")
        print(f"✅ Test AUC: {auc:.4f}")
        
        # Generate predictions and classification report
        y_pred_proba = self.model.predict(test_inputs, batch_size=batch_size, verbose=0)
        y_pred_class = (y_pred_proba > 0.5).astype(int)
        
        print("\n--- Classification Report ---")
        report = classification_report(
            y_test_final, y_pred_class, 
            target_names=['No Payment Issue (0)', 'Payment Issue (1)']
        )
        print(report)
        
        # Save model
        self.model.save('hybrid_lstm_model.h5')
        print("✅ Model saved to 'hybrid_lstm_model.h5'")
        
        return history
        
    def run_full_pipeline(self):
        """Run the complete data preparation and modeling pipeline"""
        print("=== Starting Full Data Preparation Pipeline ===")
        
        # Step 1: Load and preprocess LOS data
        self.load_los_data()
        self.preprocess_los_data()
        
        # Step 2: Load and preprocess LMS data
        self.load_lms_data()
        self.preprocess_lms_data()
        
        # Step 3: Combine datasets
        self.combine_datasets()
        
        # Step 4: Prepare features
        self.prepare_features()
        
        # Step 5: Split data
        self.split_data()
        
        # Step 6: Apply preprocessing
        self.apply_preprocessing()
        
        # Step 7: Reshape for LSTM
        self.reshape_for_lstm()
        
        # Step 8: Build model
        self.build_model()
        
        # Step 9: Train model
        history = self.train_model()
        
        print("=== Pipeline Complete ===")
        return history

if __name__ == "__main__":
    # Initialize and run the pipeline
    prep = DataPreparation()
    history = prep.run_full_pipeline()