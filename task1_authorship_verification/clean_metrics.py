import pandas as pd
import re
import ast  # For safely evaluating the dataset string dictionary

def extract_metrics(report_text):
    """Extracts precision, recall, f1-score, and support from the classification report text."""
    metrics = {
        'precision_0': None, 'recall_0': None, 'f1_score_0': None, 'support_0': None,
        'precision_1': None, 'recall_1': None, 'f1_score_1': None, 'support_1': None,
        'macro_avg_precision': None, 'macro_avg_recall': None, 'macro_avg_f1': None,
        'weighted_avg_precision': None, 'weighted_avg_recall': None, 'weighted_avg_f1': None
    }
    
    # Define regex patterns for different parts of the classification report
    class_pattern = r'\s*([01])\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9]+)'
    avg_pattern = r'(macro avg|weighted avg)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)' 
    
    for match in re.finditer(class_pattern, report_text):
        class_label, precision, recall, f1, support = match.groups()
        class_label = int(class_label)
        metrics[f'precision_{class_label}'] = float(precision)
        metrics[f'recall_{class_label}'] = float(recall)
        metrics[f'f1_score_{class_label}'] = float(f1)
        metrics[f'support_{class_label}'] = int(support)
    
    for match in re.finditer(avg_pattern, report_text):
        avg_type, precision, recall, f1 = match.groups()
        if avg_type == 'macro avg':
            metrics['macro_avg_precision'] = float(precision)
            metrics['macro_avg_recall'] = float(recall)
            metrics['macro_avg_f1'] = float(f1)
        elif avg_type == 'weighted avg':
            metrics['weighted_avg_precision'] = float(precision)
            metrics['weighted_avg_recall'] = float(recall)
            metrics['weighted_avg_f1'] = float(f1)
    
    return metrics

def extract_sampling_methods(dataset_string):
    """Extracts 'other_tweets_sampling_method' and 'user_sampling_method' from the dataset string."""
    try:
        dataset_dict = ast.literal_eval(dataset_string.replace('dataset_', '').replace('.json', ''))
        other_tweets_method = dataset_dict.get('other_tweets_sampling_method', 'unknown')
        user_method = dataset_dict.get('user_sampling_method', 'unknown')
    except (SyntaxError, ValueError):
        other_tweets_method = 'unknown'
        user_method = 'unknown'
    
    return other_tweets_method, user_method

def process_csv(input_file, output_file):
    df = pd.read_csv(input_file)
    
    # Extract sampling methods
    df[['other_tweets_sampling_method', 'user_sampling_method']] = df['dataset_file'].apply(lambda x: pd.Series(extract_sampling_methods(str(x))))
    
    # Apply extraction function to classification report column
    extracted_metrics = df['classification_report'].apply(lambda x: extract_metrics(str(x)))
    extracted_df = pd.DataFrame(extracted_metrics.tolist())
    
    # Merge with original dataframe (excluding old classification report and dataset_file)
    df = pd.concat([df.drop(columns=['dataset_file', 'classification_report']), extracted_df], axis=1)
    
    # Define the correct column order
    columns_order = ['model_name', 'other_tweets_sampling_method', 'user_sampling_method', 
                     'accuracy', 'precision', 'recall', 'f1_score',
                     'precision_0', 'recall_0', 'f1_score_0', 'support_0',
                     'precision_1', 'recall_1', 'f1_score_1', 'support_1',
                     'macro_avg_precision', 'macro_avg_recall', 'macro_avg_f1',
                     'weighted_avg_precision', 'weighted_avg_recall', 'weighted_avg_f1']
    
    # Ensure all expected columns are included in the correct order
    df = df[columns_order]

    # Save cleaned data
    df.to_csv(output_file, index=False)
    
    return df

# Run the processing
input_file = './evaluation_metrics.csv'
output_file = './cleaned_evaluation_metrics.csv'
cleaned_df = process_csv(input_file, output_file)
