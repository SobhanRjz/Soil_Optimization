import psycopg2
import logging
from typing import Dict, Any, Tuple, NamedTuple, Optional
import hashlib
from dataclasses import dataclass
import sys
from pathlib import Path

# Add parent directory to Python path to import models
sys.path.append(str(Path(__file__).parent.parent))
from models import InputData, ResultData, ParticleConfig

logger = logging.getLogger(__name__)


class PostgreSQLDatabase:
    """Handles PostgreSQL database operations for soil nailing optimization"""

    def __init__(self) -> None:
        """Initialize database connection with configuration."""
        DB_CONFIG = {
            "dbname": "soilopt",
            "user": "postgres", 
            "password": "123",
            "host": "localhost",
            "port": "5432"
        }
        self._config = DB_CONFIG
        self._conn = None
        self._cursor = None
        self._connect()

    def _connect(self) -> None:
        """Establish connection to PostgreSQL database."""
        try:
            self._conn = psycopg2.connect(**self._config)
            self._cursor = self._conn.cursor()
            logger.info("Successfully connected to PostgreSQL database")
        except psycopg2.Error as e:
            logger.error("Database connection failed: %s", e)
            raise

    def initialize_tables(self) -> None:
        """Create required database tables if they don't exist."""
        self._drop_existing_tables()
        
        try:
            self._create_inputs_table()
            self._create_results_table()
            self._conn.commit()
            logger.info("Database tables initialized successfully")
        except psycopg2.Error as e:
            logger.error("Table initialization failed: %s", e)
            raise

    def _drop_existing_tables(self) -> None:
        """Drop existing tables to start fresh."""
        self._cursor.execute("DROP TABLE IF EXISTS inputs CASCADE;")
        self._cursor.execute("DROP TABLE IF EXISTS results CASCADE;")
        self._conn.commit()
        logger.info("Existing tables dropped successfully")

    def _create_inputs_table(self) -> None:
        """Create the inputs table."""
        fields = []
        for name, field in InputData.__dataclass_fields__.items():
            if name == 'Algo_Name':
                fields.append((name, "VARCHAR(50) NOT NULL"))
            else:
                fields.append((name, "DOUBLE PRECISION NOT NULL"))
        
        columns = ",\n                ".join([
            "id SERIAL PRIMARY KEY",
            "input_hash CHAR(40) NOT NULL UNIQUE",
            "status INTEGER NOT NULL"] + 
            [f"{name} {type_}" for name, type_ in fields]
        )
        
        self._cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS inputs (
                {columns}
            );
        ''')

    def _create_results_table(self) -> None:
        """Create the results table."""
        fields = []
        for name, field in ResultData.__dataclass_fields__.items():
            fields.append((name, "DOUBLE PRECISION NOT NULL"))
            
        columns = ",\n                ".join([
            "id SERIAL PRIMARY KEY",
            "input_hash CHAR(40) NOT NULL UNIQUE",
            "status INTEGER NOT NULL"] +
            [f"{name} {type_}" for name, type_ in fields] +
            ["FOREIGN KEY(input_hash) REFERENCES inputs(input_hash) ON DELETE CASCADE"]
        )
        
        self._cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS results (
                {columns}
            );
        ''')
    def _get_result_NonCheck(self) -> str:
        """Get the input hash of one result where status is not 2.
        
        Returns:
            str: Input hash of a result with status != 2, or None if not found
        """
        self._cursor.execute('SELECT COUNT(*) FROM results WHERE status != 2 LIMIT 1')
        result = self._cursor.fetchone()
        if result:
            return result[0]
        return None
    
    def _get_result_baseHash(self, input_hash: str) -> ResultData:
        """Get the result data for a given input hash.
        
        Args:
            input_hash: The input hash to look up results for
            
        Returns:
            ResultData: The result data for the input hash, or None if not found
        """
        self._cursor.execute('SELECT * FROM results WHERE input_hash = %s ORDER BY id ASC LIMIT 1', (input_hash,))
        result = self._cursor.fetchone()
        if result:
            field_values = result[3:] # Skip id and input_hash
            result_data = ResultData(*field_values)
            return result_data
        return None
    def _get_lastHash_Result(self) -> str:
        """Get the input hash from the last row in the inputs table.
        
        Returns:
            str: The input hash from the last input, or None if table is empty
        """
        self._cursor.execute('SELECT input_hash FROM results WHERE status != 2 ORDER BY id DESC LIMIT 1')
        result = self._cursor.fetchone()
        if result:
            return result[0]
        return None
    def _get_algoNameBaseHash(self, input_hash: str) -> str:
        """Get the algorithm name for a given input hash.
        
        Args:
            input_hash: The input hash to look up the algorithm name for
            
        Returns:
            str: The algorithm name for the input hash, or None if not found
        """
        self._cursor.execute('SELECT Algo_Name FROM inputs WHERE input_hash = %s', (input_hash,))
        result = self._cursor.fetchone()
        if result:
            return result[0]
        return None
    def _get_Particle_HashBase(self, input_hash: str) -> InputData:
        """Get the particle hash base for a given input hash.
        
        Args:
            input_hash: The input hash to look up the particle hash base for
            
        Returns:
            str: The particle hash base for the input hash, or None if not found
        """
        self._cursor.execute('SELECT * FROM inputs WHERE input_hash = %s', (input_hash,))
        result = self._cursor.fetchone()
        if result:
            field_values = result[2:] # Skip id and input_hash
            result_data = InputData(*field_values)
            return result_data
        return None

        
    def insert_input(self, input_data: InputData, status: int) -> Tuple[int, str]:
        """Insert input parameters into the inputs table.
        
        Args:
            input_data: InputData containing input parameters
            status: Status code for the input
            
        Returns:
            Tuple of (input_id, input_hash)
        """
        input_dict = input_data.to_dict()
        input_hash = self._generate_hash(input_dict)

        try:
            fields = [f for f in InputData.__dataclass_fields__]
            field_names = ['input_hash', 'status'] + fields
            placeholders = ','.join(['%s'] * (len(fields) + 2))
            field_values = [input_hash, status] + [getattr(input_data, f) for f in fields]
            
            self._cursor.execute(f'''
                INSERT INTO inputs ({','.join(field_names)})
                VALUES ({placeholders})
                ON CONFLICT (input_hash) DO NOTHING
                RETURNING id;
            ''', field_values)
            input_id = self._cursor.fetchone()[0]
            self._conn.commit()
            return input_id, input_hash
            
        except psycopg2.Error as e:
            logger.error("Failed to insert input data: %s", e)
            raise

    def insert_result(self, input_hash: str, status: int, result_data: ResultData) -> None:
        """Insert optimization results into the results table.
        
        Args:
            input_hash: Hash string identifying the input parameters
            status: Status code for the result
            result_data: ResultData containing optimization results
        """
        try:
            # Update status in inputs table
            self._cursor.execute('''
                UPDATE inputs 
                SET status = %s
                WHERE input_hash = %s
            ''', (status, input_hash))

            # Insert into results table
            fields = [f for f in ResultData.__dataclass_fields__]
            field_names = ['input_hash', 'status'] + fields
            placeholders = ','.join(['%s'] * (len(fields) + 2))
            field_values = [input_hash, status] + [getattr(result_data, f) for f in fields]
            
            self._cursor.execute(f'''
                INSERT INTO results ({','.join(field_names)})
                VALUES ({placeholders})
                ON CONFLICT (input_hash) DO NOTHING
            ''', field_values)
            self._conn.commit()
        except psycopg2.Error as e:
            logger.error("Failed to insert result data: %s", e)
            raise

    def _update_result_status(self, input_hash: str, status: int) -> None:
        """Update the status field for a result record.
        
        Args:
            input_hash: Hash string identifying the input parameters
            status: New status value to set
        """
        try:
            self._cursor.execute('''
                UPDATE results 
                SET status = %s
                WHERE input_hash = %s
            ''', (status, input_hash))
            self._conn.commit()
        except psycopg2.Error as e:
            logger.error("Failed to update result status: %s", e)
            raise
    def _get_input_Data(self) -> Optional[Tuple[str, InputData]]:
        """Get one input record with status=0 from inputs table.
        
        Returns:
            Tuple of (input_hash, InputData) if found, None if no records
        """
        try:
            self._cursor.execute('''
                SELECT input_hash, rebar_dia, nail_len, nail_teta, nail_h_space, nail_v_space, algo_name
                FROM inputs 
                WHERE status = 0
                LIMIT 1
            ''')
            
            row = self._cursor.fetchone()
            if row is None:
                return None
                
            input_hash = row[0]
            input_data = InputData(
                rebar_dia=row[1],
                nail_len=row[2],
                nail_teta=row[3], 
                nail_h_space=row[4],
                nail_v_space=row[5],
                Algo_Name=row[6]
            )
            return input_hash, input_data
            
        except psycopg2.Error as e:
            logger.error("Failed to get input data: %s", e)
            raise
    @staticmethod
    def _generate_hash(data: Dict[str, Any]) -> str:
        """Generate SHA-1 hash from input data.
        
        Args:
            data: Dictionary to be hashed
            
        Returns:
            Hexadecimal hash string
        """
        hash_string = str(sorted(data.items())).encode()
        return hashlib.sha1(hash_string).hexdigest()

    def close(self) -> None:
        """Close database connection and cursor."""
        if self._cursor:
            self._cursor.close()
        if self._conn:
            self._conn.close()
            logger.info("Database connection closed")


if __name__ == "__main__":
    # Example configuration and usage
    db = PostgreSQLDatabase()
    db.initialize_tables()
    
    input_params = InputData(
        rebar_dia=10.0,
        nail_len=10.0,
        nail_teta=10.0,
        nail_h_space=10.0,
        nail_v_space=10.0,
        Algo_Name=""
    )
    
    input_id, input_hash = db.insert_input(input_params)
    db.close()
