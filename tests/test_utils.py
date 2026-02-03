import datetime
import sqlite3
import json
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

DEFAULT_DB_PATH = "tests/phase_retrieval_results.db"
RANDOM_GENERATOR = np.random.default_rng(0)


class ResultsDatabase:
    def __init__(self, db_path=DEFAULT_DB_PATH):
        self.conn = sqlite3.connect(db_path, timeout=60)
        
        # WAL mode is the 'secret sauce' for concurrency
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        
        # Increase the cache size for faster inserts
        self.conn.execute("PRAGMA cache_size=-64000;") 
        self.create_table()
    
    def create_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            case_id INTEGER,
            case_file TEXT,
            method_name TEXT,
            init_method_name TEXT,
            num_attempts INTEGER,
            error REAL,
            grad_tolerance REAL,
            max_iters INTEGER,
            fourier_oversample INTEGER,
            convergence_tolerance REAL,
            compute_duration REAL,
            near_field_md5 TEXT,
            metadata JSON,
            recovered_object BLOB
        )
        """
        self.conn.execute(query)
        self.conn.commit()

    def save_result(
        self,
        case_id: int, 
        case_file: str, 
        method_name: str, 
        init_method_name: str, 
        num_attempts: int, 
        error: float, 
        compute_duration: float, 
        grad_tolerance: float, 
        convergence_tolerance: float,
        max_iters: int,
        fourier_oversample: int,
        near_field_md5: str,
        metadata: dict,
        recovered_object: jnp.ndarray = None
    ):
        query = """
            
        INSERT INTO results (timestamp, case_id, case_file, method_name, init_method_name, num_attempts, error, grad_tolerance, max_iters, fourier_oversample, convergence_tolerance, compute_duration, near_field_md5, metadata, recovered_object)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.conn.execute(query, (
            datetime.datetime.now().isoformat(),
            case_id, 
            case_file, 
            method_name, 
            init_method_name, 
            num_attempts, 
            float(error), 
            float(grad_tolerance), 
            max_iters, 
            fourier_oversample, 
            float(convergence_tolerance), 
            float(compute_duration), 
            near_field_md5, 
            json.dumps(metadata),
            recovered_object.tobytes() if recovered_object is not None else None
        ))
        self.conn.commit()

    def close(self):
        self.conn.close()


class CachedGenerator:
    def __init__(self, iterable):
        self.iterable = iter(iterable)
        self.cache = []

    def __iter__(self):
        # First, yield everything we've already computed
        yield from self.cache
        
        # Then, continue computing new values and caching them
        for item in self.iterable:
            self.cache.append(item)
            yield item


def align_global_phase(x_true, x_rec):
    # Calculate the inner product (complex)
    inner_product = jnp.sum(jnp.conj(x_true) * x_rec)
    # Get the phase of that inner product
    phase_shift = jnp.sign(inner_product)
    # Rotate the reconstruction to match the truth
    return x_rec / phase_shift

