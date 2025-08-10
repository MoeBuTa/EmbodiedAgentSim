from typing import Optional, Dict
import pandas as pd
from datetime import datetime
from easim.utils.constants import EVALUATION_DIR



def save_evaluation_results(task_name: str, metrics: Dict[str, float], num_episodes: int, agent_name: str = "unknown", agent_model: str = "unknown") -> None:
    """
    Append evaluation results to a consolidated CSV file.
    
    :param task_name: Name of the task being evaluated
    :param metrics: Dictionary of evaluation metrics
    :param num_episodes: Number of episodes evaluated
    :param agent_name: Name/type of the agent being evaluated
    :param agent_model: Model used by the agent (e.g., gpt-4o-mini)
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = "evaluation_results.csv"
    filepath = EVALUATION_DIR / filename
    
    # Prepare data for CSV
    data = {
        'task_name': [task_name],
        'agent_name': [agent_name],
        'agent_model': [agent_model],
        'num_episodes': [num_episodes],
        'timestamp': [timestamp]
    }
    
    # Add all metrics as columns
    for metric_name, metric_value in metrics.items():
        data[metric_name] = [metric_value]
    
    # Create DataFrame for new row
    new_row = pd.DataFrame(data)
    
    # Check if file exists and append or create
    if filepath.exists():
        # Read existing data to get all columns
        existing_df = pd.read_csv(filepath)
        # Combine with new row, filling missing columns with NaN
        combined_df = pd.concat([existing_df, new_row], ignore_index=True, sort=False)
        combined_df.to_csv(filepath, index=False)
    else:
        # Create new file
        new_row.to_csv(filepath, index=False)
