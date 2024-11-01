import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self):
        """Initialize data storage structure"""
        # Create base directories
        self.base_dir = Path("data")
        self.datasets_dir = self.base_dir / "datasets"
        self.metadata_dir = self.base_dir / "metadata"
        self.processed_dir = self.base_dir / "processed"
        
        # Create directory structure
        self._create_directory_structure()
        
        # Initialize metadata tracking
        self.metadata_file = self.metadata_dir / "dataset_metadata.json"
        self.metadata = self._load_metadata()
    
    def _create_directory_structure(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.base_dir,
            self.datasets_dir,
            self.metadata_dir,
            self.processed_dir,
            self.datasets_dir / "medical",
            self.datasets_dir / "environmental",
            self.datasets_dir / "financial",
            self.datasets_dir / "social"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def _load_metadata(self) -> Dict:
        """Load dataset metadata from JSON file"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save metadata to JSON file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=4)
    
    def add_dataset(self, 
                   name: str, 
                   category: str, 
                   data: pd.DataFrame, 
                   description: str = "",
                   source: str = "",
                   tags: List[str] = None) -> bool:
        """
        Add a new dataset to storage
        
        Args:
            name: Dataset name
            category: Dataset category (medical, environmental, etc.)
            data: Pandas DataFrame containing the dataset
            description: Dataset description
            source: Data source information
            tags: List of relevant tags
        """
        try:
            # Create dataset ID and path
            dataset_id = f"{category}_{name}_{datetime.now().strftime('%Y%m%d')}"
            dataset_path = self.datasets_dir / category / f"{dataset_id}.csv"
            
            # Save dataset
            data.to_csv(dataset_path, index=False)
            
            # Update metadata
            self.metadata[dataset_id] = {
                "name": name,
                "category": category,
                "description": description,
                "source": source,
                "tags": tags or [],
                "created_date": datetime.now().isoformat(),
                "rows": len(data),
                "columns": list(data.columns),
                "file_path": str(dataset_path)
            }
            
            # Save updated metadata
            self._save_metadata()
            
            logger.info(f"Successfully added dataset: {dataset_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding dataset {name}: {str(e)}")
            return False
    
    def get_dataset(self, dataset_id: str) -> pd.DataFrame:
        """Retrieve a dataset by ID"""
        try:
            if dataset_id in self.metadata:
                file_path = Path(self.metadata[dataset_id]["file_path"])
                return pd.read_csv(file_path)
            else:
                logger.warning(f"Dataset not found: {dataset_id}")
                return None
        except Exception as e:
            logger.error(f"Error retrieving dataset {dataset_id}: {str(e)}")
            return None
    
    def list_datasets(self, category: str = None) -> List[Dict]:
        """List available datasets, optionally filtered by category"""
        datasets = []
        for dataset_id, metadata in self.metadata.items():
            if category is None or metadata["category"] == category:
                datasets.append({
                    "id": dataset_id,
                    "name": metadata["name"],
                    "category": metadata["category"],
                    "description": metadata["description"],
                    "rows": metadata["rows"],
                    "created_date": metadata["created_date"]
                })
        return datasets
    
    def get_dataset_info(self, dataset_id: str) -> Dict:
        """Get detailed information about a dataset"""
        return self.metadata.get(dataset_id)
    
    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset and its metadata"""
        try:
            if dataset_id in self.metadata:
                # Remove file
                file_path = Path(self.metadata[dataset_id]["file_path"])
                if file_path.exists():
                    file_path.unlink()
                
                # Remove metadata
                del self.metadata[dataset_id]
                self._save_metadata()
                
                logger.info(f"Successfully deleted dataset: {dataset_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting dataset {dataset_id}: {str(e)}")
            return False 