import pandas as pd
from pathlib import Path
from data_storage.data_manager import DataManager
import os

def create_directory_structure():
    """Create necessary directories for data storage"""
    base_dir = Path("data")
    directories = [
        base_dir,
        base_dir / "datasets",
        base_dir / "datasets" / "hr_analytics",
        base_dir / "metadata",
        base_dir / "processed"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def import_hr_dataset(file_path: str):
    """
    Import the HR dataset into the data storage system
    """
    try:
        # Create directory structure first
        create_directory_structure()
        
        # Initialize data manager
        data_manager = DataManager()
        
        # Load the HR dataset
        print(f"Loading HR dataset from: {file_path}")
        dataset = pd.read_csv(file_path)
        
        # Add dataset to storage
        success = data_manager.add_dataset(
            name="HRDataset_v14",
            category="hr_analytics",
            data=dataset,
            description="Human Resources Analytics Dataset Version 14",
            source="HR Analytics Dataset",
            tags=["HR", "analytics", "employee_data", "training"]
        )
        
        if success:
            print("\nSuccessfully imported HR Dataset")
            print("\nDataset Summary:")
            print(f"- Total Records: {len(dataset)}")
            print(f"- Features: {list(dataset.columns)}")
            print(f"- Memory Usage: {dataset.memory_usage().sum() / 1024**2:.2f} MB")
            
            # Display first few rows
            print("\nFirst few rows of the dataset:")
            print(dataset.head())
            
        else:
            print("Failed to import dataset")
            
    except FileNotFoundError:
        print(f"Error: Could not find the file at {file_path}")
        print("Please check if the file exists at this location")
    except Exception as e:
        print(f"Error importing dataset: {str(e)}")

if __name__ == "__main__":
    # Path to the HR dataset in Downloads folder
    dataset_path = "C:/Users/leoco/Downloads/HRDataset_v14.csv"
    
    # Check if file exists
    if Path(dataset_path).exists():
        print(f"Found dataset at: {dataset_path}")
        import_hr_dataset(dataset_path)
    else:
        print("Could not find HRDataset_v14.csv in Downloads folder")
        print("Please enter the correct path to the CSV file:")
        user_path = input("Path: ").strip('"')
        if Path(user_path).exists():
            import_hr_dataset(user_path)
        else:
            print("File not found. Please check the file location and try again.")