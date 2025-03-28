# Standard library imports
import os
import sqlite3
import threading
import time
import signal
import sys
from functools import wraps

# Third-party imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm, anderson, pearsonr, probplot, iqr, zscore
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Local imports
from src.api_model import EragAPI
from src.settings import settings
from src.look_and_feel import error, success, warning, info, highlight
from src.print_pdf import PDFReportGenerator
from src.helper_da import get_technique_info

class TimeoutException(Exception):
    pass

class ExploratoryDataAnalysis:
    def __init__(self, worker_erag_api, supervisor_erag_api, db_path):
        self.worker_erag_api = worker_erag_api
        self.supervisor_erag_api = supervisor_erag_api
        self.db_path = db_path
        self.technique_counter = 1
        self.total_techniques = 10  # Updated to reflect merged methods
        self.table_name = None
        self.output_folder = None
        self.text_output = ""
        self.pdf_content = []
        self.findings = []
        self.llm_name = f"Worker: {self.worker_erag_api.model}, Supervisor: {self.supervisor_erag_api.model}"
        self.toc_entries = []
        self.image_paths = []
        self.max_pixels = 400000
        self.timeout_seconds = 10
        self.image_data = []
        self.pdf_generator = None
        self.settings = settings
        self.database_description = ""
        self.paused = False
        self.setup_signal_handler()

    def setup_signal_handler(self):
        """Set up signal handler for Ctrl+C"""
        signal.signal(signal.SIGINT, self.handle_signal)

    def handle_signal(self, sig, frame):
        """Handle Ctrl+C by pausing execution"""
        if not self.paused:
            self.paused = True
            print(warning("\nScript paused. Press Enter to continue or Ctrl+C again to exit..."))
            try:
                user_input = input()
                self.paused = False
                print(info("Resuming execution..."))
            except KeyboardInterrupt:
                print(error("\nExiting script..."))
                sys.exit(0)
        else:
            print(error("\nExiting script..."))
            sys.exit(0)

    def check_if_paused(self):
        """Check if execution is paused and wait for Enter if needed"""
        while self.paused:
            time.sleep(0.1)  # Small sleep to prevent CPU hogging

    def calculate_figure_size(self, aspect_ratio=16/9):
        max_width = int(np.sqrt(self.max_pixels * aspect_ratio))
        max_height = int(max_width / aspect_ratio)
        return (max_width / 100, max_height / 100)

    def timeout(timeout_duration):
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                result = [TimeoutException("Function call timed out")]

                def target():
                    try:
                        result[0] = func(self, *args, **kwargs)
                    except Exception as e:
                        result[0] = e

                thread = threading.Thread(target=target)
                thread.start()
                thread.join(timeout_duration)

                if thread.is_alive():
                    print(f"Warning: {func.__name__} timed out after {timeout_duration} seconds. Skipping this graphic.")
                    return None
                else:
                    if isinstance(result[0], Exception):
                        raise result[0]
                    return result[0]
            return wrapper
        return decorator

    @timeout(10)
    def generate_plot(self, plot_function, *args, **kwargs):
        return plot_function(*args, **kwargs)

    def prompt_for_database_description(self):
        """Ask the user for a description of the database"""
        print(info("Please provide a description of the database. This will help the AI models provide better insights."))
        print(info("Describe the purpose, main tables, key data points, and any other relevant information:"))
        self.database_description = input("> ")
        print(success(f"Database description received: {self.database_description}"))

    def run(self):
        self.prompt_for_database_description()
        print(info(f"Starting Exploratory Data Analysis on {self.db_path}"))
        tables = self.get_tables()
        for table in tables:
            self.analyze_table(table)
        
        self.save_text_output()
        self.generate_pdf_report()
        print(success(f"Exploratory Data Analysis completed. Results saved in {self.output_folder}"))

    def get_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            return [table[0] for table in cursor.fetchall()]

    def analyze_table(self, table_name):
        self.table_name = table_name
        self.output_folder = os.path.join(settings.output_folder, f"xda_{self.table_name}")
        os.makedirs(self.output_folder, exist_ok=True)
        
        self.pdf_generator = PDFReportGenerator(self.output_folder, self.llm_name, self.table_name)
        
        print(highlight(f"\nAnalyzing table: {table_name}"))
        self.text_output += f"\nAnalyzing table: {table_name}\n"
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            print(info(f"Loaded full dataset with {len(df)} rows and {len(df.columns)} columns"))

        analysis_methods = [
            self.overall_table_analysis,
            self.statistical_analysis,
            self.correlation_analysis,
            self.categorical_features_analysis,
            self.distribution_analysis,
            self.outlier_detection,
            self.time_series_analysis,
            self.feature_importance_analysis,
            self.dimensionality_reduction_analysis,
            self.cluster_analysis
        ]

        for method in analysis_methods:
            try:
                # Check if execution is paused
                self.check_if_paused()
                method(df, table_name)
            except Exception as e:
                error_message = f"An error occurred during {method.__name__}: {str(e)}"
                print(error(error_message))
                self.text_output += f"\n{error_message}\n"
                self.pdf_content.append((method.__name__, [], error_message))
                
                # Write error to method-specific output file
                method_name = method.__name__
                with open(os.path.join(self.output_folder, f"{method_name}_results.txt"), "w", encoding='utf-8') as f:
                    f.write(error_message)

    def overall_table_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Overall Table Analysis"))
        
        # Extract actual table data for context
        column_types = df.dtypes.value_counts()
        column_type_names = [f"{index} ({count} columns)" for index, count in column_types.items()]
        
        results = {
            "Total Rows": len(df),
            "Total Columns": len(df.columns),
            "Column Types": df.dtypes.value_counts().to_dict(),
            "Column Type Names": column_type_names,
            "Memory Usage": df.memory_usage(deep=True).sum() / 1024**2,  # in MB
            "Column Names": df.columns.tolist(),
            "Missing Values": df.isnull().sum().to_dict(),
            "Missing Percentages": (df.isnull().sum() / len(df) * 100).to_dict(),
            "Unique Values": df.nunique().to_dict()
        }
        
        def plot_overall_description():
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(self.calculate_figure_size()[0]*2, self.calculate_figure_size()[1]*2))
            
            # Column types pie chart
            column_types = df.dtypes.value_counts()
            ax1.pie(column_types.values, labels=column_types.index, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Distribution of Column Types')
            
            # Data completeness
            completeness = 1 - (df.isnull().sum() / len(df))
            completeness = completeness.sort_values(ascending=False)
            ax2.bar(completeness.index, completeness.values)
            ax2.set_title('Data Completeness by Column')
            ax2.set_xlabel('Columns')
            ax2.set_ylabel('Completeness (%)')
            ax2.set_ylim(0, 1)
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90, ha='right')
            
            # Unique values
            unique_counts = df.nunique().sort_values(ascending=False)
            ax3.bar(unique_counts.index, unique_counts.values)
            ax3.set_title('Unique Values by Column')
            ax3.set_xlabel('Columns')
            ax3.set_ylabel('Unique Value Count')
            ax3.set_xticklabels(ax3.get_xticklabels(), rotation=90, ha='right')
            
            # Data type distribution pie chart
            data_type_counts = df.dtypes.map(lambda x: x.name).value_counts()
            ax4.pie(data_type_counts.values, labels=data_type_counts.index, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Distribution of Data Types')
            
            plt.tight_layout()
            return fig, (ax1, ax2, ax3, ax4)

        result = self.generate_plot(plot_overall_description)
        if result is not None:
            fig, _ = result
            img_path = os.path.join(self.output_folder, f"{table_name}_overall_analysis.png")
            plt.savefig(img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            results['image_paths'] = [("Overall Table Analysis", img_path)]
        
        self.interpret_results("Overall Table Analysis", results, table_name)
        self.technique_counter += 1

    def statistical_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Statistical Analysis"))

        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        
        # Extract actual column statistics for better context
        results = {
            "table_overview": {
                "total_rows": len(df),
                "numeric_columns": len(numeric_columns),
                "column_names": numeric_columns.tolist()
            }
        }
        
        # Include descriptive statistics for each column
        for col in numeric_columns:
            results[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurtosis(),
                'missing_values': df[col].isna().sum(),
                'missing_percentage': (df[col].isna().sum() / len(df)) * 100
            }

        # Generate plots for each column
        image_paths = []
        
        for col in numeric_columns:
            avg = df[col].mean()
            median = df[col].median()
            std_dev = df[col].std()
            min_val = df[col].min()
            max_val = df[col].max()
            
            fig, ax = plt.subplots(figsize=self.calculate_figure_size())
            
            measures = ['Average', 'Median', 'Std Dev', 'Min', 'Max']
            values = [avg, median, std_dev, min_val, max_val]
            colors = ['b', 'g', 'r', 'c', 'orange']
            
            bars = ax.bar(measures, values, color=colors, alpha=0.7)
            
            # Annotate the bars with the respective values
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{value:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            ax.set_ylabel('Value')
            ax.set_title(f'Statistical Measures for {col}')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            img_path = os.path.join(self.output_folder, f"{table_name}_{col}_statistical_measures.png")
            plt.savefig(img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            image_paths.append((f"Statistical Measures for {col}", img_path))
        
        results['image_paths'] = image_paths
        
        self.interpret_results("Statistical Analysis", results, table_name)
        self.technique_counter += 1

    def correlation_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Correlation Analysis"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        results = {}
        image_paths = []
        
        if len(numerical_columns) > 1:
            correlation_matrix = df[numerical_columns].corr()
            
            # Extract actual correlation pairs for context
            top_correlations = []
            for i in range(len(numerical_columns)):
                for j in range(i+1, len(numerical_columns)):
                    col1 = numerical_columns[i]
                    col2 = numerical_columns[j]
                    corr_value = correlation_matrix.loc[col1, col2]
                    if abs(corr_value) > 0.5:  # Only include significant correlations
                        top_correlations.append({
                            'column_1': col1,
                            'column_2': col2,
                            'correlation': corr_value
                        })
            
            # Sort by absolute correlation value
            top_correlations = sorted(top_correlations, key=lambda x: abs(x['correlation']), reverse=True)
            results['significant_correlations'] = top_correlations
            
            # Function to create correlation heatmap
            def plot_correlation_heatmap(corr_matrix, title):
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, ax=ax)
                ax.set_title(title)
                return fig, ax
            
            # Create full correlation matrix
            result = self.generate_plot(plot_correlation_heatmap, correlation_matrix, 'Full Correlation Matrix')
            if result is not None:
                fig, ax = result
                img_path = os.path.join(self.output_folder, f"{table_name}_full_correlation_matrix.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(("Full Correlation Matrix", img_path))
            
            # Create correlation matrix for highly correlated features (threshold 0.5)
            high_corr_threshold = 0.5
            high_corr = (correlation_matrix.abs() > high_corr_threshold) & (correlation_matrix != 1.0)
            high_corr_features = high_corr.any().index[high_corr.any()]
            
            if len(high_corr_features) > 1:
                high_corr_matrix = correlation_matrix.loc[high_corr_features, high_corr_features]
                result = self.generate_plot(plot_correlation_heatmap, high_corr_matrix, f'High Correlation Matrix (|r| > {high_corr_threshold})')
                if result is not None:
                    fig, ax = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_high_correlation_matrix_0.5.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append((f"High Correlation Matrix (|r| > {high_corr_threshold})", img_path))
            
            # Create correlation matrix for very highly correlated features (threshold 0.75)
            very_high_corr_threshold = 0.75
            very_high_corr = (correlation_matrix.abs() > very_high_corr_threshold) & (correlation_matrix != 1.0)
            very_high_corr_features = very_high_corr.any().index[very_high_corr.any()]
            
            if len(very_high_corr_features) > 1:
                very_high_corr_matrix = correlation_matrix.loc[very_high_corr_features, very_high_corr_features]
                result = self.generate_plot(plot_correlation_heatmap, very_high_corr_matrix, f'Very High Correlation Matrix (|r| > {very_high_corr_threshold})')
                if result is not None:
                    fig, ax = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_high_correlation_matrix_0.75.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append((f"Very High Correlation Matrix (|r| > {very_high_corr_threshold})", img_path))
            
            # If there are many features, create subset correlation matrices
            max_features_per_plot = 10
            if len(numerical_columns) > max_features_per_plot:
                for i in range(0, len(numerical_columns), max_features_per_plot):
                    subset_columns = numerical_columns[i:i+max_features_per_plot]
                    subset_corr = correlation_matrix.loc[subset_columns, subset_columns]
                    result = self.generate_plot(plot_correlation_heatmap, subset_corr, f'Correlation Matrix (Subset {i//max_features_per_plot + 1})')
                    if result is not None:
                        fig, ax = result
                        img_path = os.path.join(self.output_folder, f"{table_name}_correlation_matrix_subset_{i//max_features_per_plot + 1}.png")
                        plt.savefig(img_path, dpi=100, bbox_inches='tight')
                        plt.close(fig)
                        image_paths.append((f"Correlation Matrix (Subset {i//max_features_per_plot + 1})", img_path))
            
            # Store correlation values
            results['correlation_matrix'] = correlation_matrix.to_dict()
            
            # Find top positive and negative correlations
            correlations = correlation_matrix.unstack()
            correlations = correlations[correlations != 1.0]  # Remove self-correlations
            top_positive = correlations.nlargest(5)
            top_negative = correlations.nsmallest(5)
            
            results['top_positive_correlations'] = [
                {"pair": f"{idx[0]} ↔ {idx[1]}", "value": val}
                for idx, val in top_positive.items()
            ]
            
            results['top_negative_correlations'] = [
                {"pair": f"{idx[0]} ↔ {idx[1]}", "value": val}
                for idx, val in top_negative.items()
            ]
            
        else:
            results = "N/A - Not enough numerical features for correlation analysis"
        
        results['image_paths'] = image_paths
        self.interpret_results("Correlation Analysis", results, table_name)
        self.technique_counter += 1

    def categorical_features_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Categorical Features Analysis"))
        
        categorical_columns = df.select_dtypes(include=['object']).columns
        image_paths = []
        results = {}
        
        if len(categorical_columns) > 0:
            # Include actual category names and values
            for col in categorical_columns:
                value_counts = df[col].value_counts()
                # Get actual names of categories (not coded as Alpha, Beta, etc.)
                if len(value_counts) > 10:
                    top_9 = value_counts.nlargest(9)
                    other = value_counts.iloc[9:].sum()
                    categories = top_9.index.tolist()
                    categories.append('Other')
                    counts = top_9.values.tolist()
                    counts.append(other)
                else:
                    categories = value_counts.index.tolist()
                    counts = value_counts.values.tolist()
                
                results[col] = {
                    'total_categories': len(value_counts),
                    'top_categories': [
                        {"name": cat, "count": count, "percentage": (count/len(df))*100} 
                        for cat, count in zip(categories, counts)
                    ],
                    'null_count': df[col].isna().sum(),
                    'null_percentage': (df[col].isna().sum() / len(df)) * 100
                }
                
                def plot_categorical_distribution():
                    if len(value_counts) > 10:
                        top_10 = pd.Series(counts, index=categories)
                    else:
                        top_10 = value_counts.nlargest(10)
                        
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
                    
                    # Bar plot
                    sns.barplot(x=top_10.index, y=top_10.values, ax=ax1, palette='Set3')
                    ax1.set_title(f'Top Categories in {col}')
                    ax1.set_xlabel(col)
                    ax1.set_ylabel('Count')
                    ax1.tick_params(axis='x', rotation=45)
                    for i, v in enumerate(top_10.values):
                        ax1.text(i, v, str(v), ha='center', va='bottom')
                    
                    # Pie chart
                    ax2.pie(top_10.values, labels=top_10.index, autopct='%1.1f%%', startangle=90)
                    ax2.set_title(f'Top Categories Distribution in {col}')
                    
                    plt.tight_layout()
                    return fig, (ax1, ax2)

                result = self.generate_plot(plot_categorical_distribution)
                if result is not None:
                    fig, _ = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_{col}_categorical_distribution.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append((f"{col} Categorical Distribution", img_path))
            
            results['image_paths'] = image_paths
        else:
            results = "N/A - No categorical features found"
        
        self.interpret_results("Categorical Features Analysis", results, table_name)
        self.technique_counter += 1

    def distribution_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Distribution Analysis"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        image_paths = []
        results = {}
        
        for col in numerical_columns:
            # Calculate distribution statistics with actual column name
            col_data = df[col].dropna()
            distribution_stats = {
                'mean': col_data.mean(),
                'median': col_data.median(),
                'std': col_data.std(),
                'skewness': col_data.skew(),
                'kurtosis': col_data.kurtosis()
            }
            
            # Run normality test
            try:
                anderson_result = anderson(col_data)
                distribution_stats['anderson_darling_statistic'] = anderson_result.statistic
                distribution_stats['anderson_critical_values'] = anderson_result.critical_values.tolist()
                distribution_stats['anderson_significance_levels'] = [15.0, 10.0, 5.0, 2.5, 1.0]
                # Determine if normally distributed
                if anderson_result.statistic < anderson_result.critical_values[2]:  # Using 5% significance level
                    distribution_stats['is_normal'] = True
                else:
                    distribution_stats['is_normal'] = False
            except:
                distribution_stats['anderson_test'] = "Failed to compute"
                
            results[col] = distribution_stats
            
            def plot_distribution():
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
                
                # Histogram with KDE
                sns.histplot(df[col], kde=True, ax=ax1)
                ax1.set_title(f'Distribution of {col}')
                ax1.set_xlabel(col)
                ax1.set_ylabel('Frequency')
                
                # Q-Q plot
                probplot(df[col], dist="norm", plot=ax2)
                ax2.set_title(f'Q-Q Plot of {col}')
                
                plt.tight_layout()
                return fig, (ax1, ax2)

            result = self.generate_plot(plot_distribution)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_distribution.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append((f"{col} Distribution", img_path))
        
        results['image_paths'] = image_paths
        self.interpret_results("Distribution Analysis", results, table_name)
        self.technique_counter += 1

    def outlier_detection(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Outlier Detection"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        results = {}
        image_paths = []
        
        for col in numerical_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            
            # Get actual outlier values (not just statistics)
            if len(outliers) > 0:
                top_outliers = outliers.nlargest(min(5, len(outliers)))
                bottom_outliers = outliers.nsmallest(min(5, len(outliers)))
                
                outlier_values = {
                    'top_high_outliers': [
                        {"value": float(value), "z_score": float(zscore(df[col])[i])}
                        for i, value in enumerate(top_outliers)
                    ],
                    'top_low_outliers': [
                        {"value": float(value), "z_score": float(zscore(df[col])[i])}
                        for i, value in enumerate(bottom_outliers)
                    ]
                }
            else:
                outlier_values = {'top_high_outliers': [], 'top_low_outliers': []}
            
            results[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(df)) * 100,
                'range': (float(lower_bound), float(upper_bound)),
                'outlier_values': outlier_values,
                'summary': {
                    'Q1': float(Q1),
                    'median': float(df[col].median()),
                    'Q3': float(Q3),
                    'IQR': float(IQR),
                    'min_non_outlier': float(df[(df[col] >= lower_bound) & (df[col] <= upper_bound)][col].min()) if len(df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]) > 0 else None,
                    'max_non_outlier': float(df[(df[col] >= lower_bound) & (df[col] <= upper_bound)][col].max()) if len(df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]) > 0 else None
                }
            }
            
            def plot_boxplot():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                sns.boxplot(x=df[col], ax=ax)
                ax.set_title(f'Box Plot of {col}')
                ax.set_xlabel(col)
                return fig, ax

            result = self.generate_plot(plot_boxplot)
            if result is not None:
                fig, ax = result
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_boxplot.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append((f"{col} Box Plot", img_path))
        
        results['image_paths'] = image_paths
        self.interpret_results("Outlier Detection", results, table_name)
        self.technique_counter += 1

    def time_series_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Time Series Analysis"))
        
        image_paths = []
        results = {}
        
        # Try to identify date columns
        date_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # If no datetime columns found, try to convert string columns to datetime
        if not date_columns:
            for col in df.select_dtypes(include=['object']):
                try:
                    df[col] = pd.to_datetime(df[col])
                    date_columns.append(col)
                except ValueError:
                    continue
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        
        if date_columns and len(numerical_columns) > 0:
            date_col = date_columns[0]  # Use the first identified date column
            df = df.sort_values(date_col)  # Sort by date
            
            # Extract time series insights
            results['time_series_overview'] = {
                'date_column': date_col,
                'start_date': df[date_col].min().strftime('%Y-%m-%d') if hasattr(df[date_col].min(), 'strftime') else str(df[date_col].min()),
                'end_date': df[date_col].max().strftime('%Y-%m-%d') if hasattr(df[date_col].max(), 'strftime') else str(df[date_col].max()),
                'time_span_days': (df[date_col].max() - df[date_col].min()).days if hasattr(df[date_col].max(), 'strftime') else None,
                'number_of_periods': len(df),
                'numerical_columns': numerical_columns.tolist()
            }
            
            for num_col in numerical_columns:
                # Calculate time series metrics
                column_metrics = {
                    'trend': None,  # Will be filled by AI interpretation
                    'min_value': float(df[num_col].min()),
                    'max_value': float(df[num_col].max()),
                    'mean': float(df[num_col].mean()),
                    'std_dev': float(df[num_col].std())
                }
                
                # Add growth metrics if possible
                if len(df) > 1:
                    first_value = df[num_col].iloc[0]
                    last_value = df[num_col].iloc[-1]
                    if first_value != 0:
                        column_metrics['overall_growth_pct'] = ((last_value - first_value) / first_value) * 100
                    
                results[num_col] = column_metrics
                
                def plot_time_series():
                    fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                    ax.plot(df[date_col], df[num_col])
                    ax.set_title(f'Time Series Plot of {num_col}')
                    ax.set_xlabel('Date')
                    ax.set_ylabel(num_col)
                    plt.xticks(rotation=45)
                    fig.tight_layout()
                    return fig, ax

                result = self.generate_plot(plot_time_series)
                if result is not None:
                    fig, ax = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_{num_col}_time_series_plot.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append((f"{num_col} Time Series", img_path))
            
            results['date_column_used'] = date_col
            results['numerical_columns_analyzed'] = numerical_columns.tolist()
            results['image_paths'] = image_paths
        else:
            if not date_columns:
                results['error'] = "No suitable date columns found for time series analysis."
            elif len(numerical_columns) == 0:
                results['error'] = "No numerical columns found for time series analysis."
            else:
                results['error'] = "Insufficient data for time series analysis."
        
        self.interpret_results("Time Series Analysis", results, table_name)
        self.technique_counter += 1

    def feature_importance_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Feature Importance Analysis"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        results = {}
        
        if len(numerical_columns) > 1:
            X = df[numerical_columns].drop(numerical_columns[-1], axis=1)
            y = df[numerical_columns[-1]]
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Include actual feature names and importance scores
            results['target_variable'] = numerical_columns[-1]
            results['features_ranked'] = [
                {"feature": row['feature'], "importance": row['importance']} 
                for _, row in feature_importance.iterrows()
            ]
            
            def plot_feature_importance():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
                ax.set_title('Feature Importance')
                ax.set_xlabel('Importance')
                ax.set_ylabel('Feature')
                return fig, ax

            result = self.generate_plot(plot_feature_importance)
            if result is not None:
                fig, ax = result
                img_path = os.path.join(self.output_folder, f"{table_name}_feature_importance.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                results['image_paths'] = [("Feature Importance", img_path)]
        else:
            results = "N/A - Not enough numerical columns for feature importance analysis"
        
        self.interpret_results("Feature Importance Analysis", results, table_name)
        self.technique_counter += 1

    def dimensionality_reduction_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Dimensionality Reduction Analysis"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        results = {}
        
        if len(numerical_columns) > 2:
            X_scaled = StandardScaler().fit_transform(df[numerical_columns])
            
            pca = PCA()
            pca_result = pca.fit_transform(X_scaled)
            
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
            
            # Include actual variance explained by components
            results['total_components'] = len(explained_variance_ratio)
            results['explained_variance_by_component'] = [
                {"component": f"PC{i+1}", "variance_explained": var, "cumulative_variance": cum_var}
                for i, (var, cum_var) in enumerate(zip(explained_variance_ratio, cumulative_variance_ratio))
            ]
            
            # Calculate number of components for 80% and 90% variance
            components_80pct = next((i+1 for i, cum_var in enumerate(cumulative_variance_ratio) if cum_var >= 0.8), len(cumulative_variance_ratio))
            components_90pct = next((i+1 for i, cum_var in enumerate(cumulative_variance_ratio) if cum_var >= 0.9), len(cumulative_variance_ratio))
            
            results['components_for_80pct_variance'] = components_80pct
            results['components_for_90pct_variance'] = components_90pct
            results['dimensionality_reduction_potential'] = {
                'original_dimensions': len(numerical_columns),
                'recommended_dimensions': components_80pct,
                'reduction_percentage': ((len(numerical_columns) - components_80pct) / len(numerical_columns)) * 100
            }
            
            def plot_pca():
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.calculate_figure_size()[0]*2, self.calculate_figure_size()[1]))
                
                # Scree plot
                ax1.plot(range(1, len(explained_variance_ratio)+1), explained_variance_ratio, 'bo-')
                ax1.set_xlabel('Principal Component')
                ax1.set_ylabel('Explained Variance Ratio')
                ax1.set_title('Scree Plot')
                
                # Cumulative explained variance plot
                ax2.plot(range(1, len(cumulative_variance_ratio)+1), cumulative_variance_ratio, 'ro-')
                ax2.set_xlabel('Number of Components')
                ax2.set_ylabel('Cumulative Explained Variance Ratio')
                ax2.set_title('Cumulative Explained Variance')
                
                plt.tight_layout()
                return fig, (ax1, ax2)

            result = self.generate_plot(plot_pca)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_pca_analysis.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                results['image_paths'] = [("PCA Analysis", img_path)]
        else:
            results = "N/A - Not enough numerical columns for PCA analysis"
        
        self.interpret_results("Dimensionality Reduction Analysis", results, table_name)
        self.technique_counter += 1

    def cluster_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Cluster Analysis"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        results = {}
        
        if len(numerical_columns) > 1:
            X_scaled = StandardScaler().fit_transform(df[numerical_columns])
            
            # Determine optimal number of clusters using elbow method
            inertias = []
            max_clusters = min(10, X_scaled.shape[0] - 1)
            for k in range(1, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(X_scaled)
                inertias.append(kmeans.inertia_)
            
            # Find elbow point
            elbow = next(i for i in range(1, len(inertias)) if inertias[i-1] - inertias[i] < (inertias[0] - inertias[-1]) / 10)
            
            results['optimal_clusters'] = elbow
            results['inertias'] = inertias
            
            # Perform K-means clustering with optimal number of clusters
            kmeans = KMeans(n_clusters=elbow, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Analyze clusters
            df_with_clusters = df.copy()
            df_with_clusters['cluster'] = cluster_labels
            
            # Get actual descriptions of each cluster
            cluster_profiles = []
            for cluster_id in range(elbow):
                cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
                
                cluster_profile = {
                    'cluster_id': cluster_id,
                    'size': len(cluster_data),
                    'percentage': (len(cluster_data) / len(df)) * 100,
                    'feature_means': {}
                }
                
                # Calculate mean for each feature in this cluster
                for col in numerical_columns:
                    cluster_mean = cluster_data[col].mean()
                    overall_mean = df[col].mean()
                    difference_pct = ((cluster_mean - overall_mean) / overall_mean) * 100 if overall_mean != 0 else 0
                    
                    cluster_profile['feature_means'][col] = {
                        'cluster_mean': float(cluster_mean),
                        'overall_mean': float(overall_mean),
                        'difference_pct': float(difference_pct)
                    }
                
                cluster_profiles.append(cluster_profile)
            
            results['cluster_profiles'] = cluster_profiles
            
            def plot_clusters():
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.calculate_figure_size()[0]*2, self.calculate_figure_size()[1]))
                
                # Elbow plot
                ax1.plot(range(1, max_clusters + 1), inertias, 'bo-')
                ax1.set_xlabel('Number of Clusters (k)')
                ax1.set_ylabel('Inertia')
                ax1.set_title('Elbow Method for Optimal k')
                ax1.axvline(x=elbow, color='r', linestyle='--')
                
                # 2D projection of clusters
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis')
                ax2.set_xlabel('First Principal Component')
                ax2.set_ylabel('Second Principal Component')
                ax2.set_title(f'2D PCA Projection of {elbow} Clusters')
                
                plt.tight_layout()
                return fig, (ax1, ax2)

            result = self.generate_plot(plot_clusters)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_cluster_analysis.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                results['image_paths'] = [("Cluster Analysis", img_path)]
        else:
            results = "N/A - Not enough numerical columns for cluster analysis"
        
        self.interpret_results("Cluster Analysis", results, table_name)
        self.technique_counter += 1

    def save_results(self, analysis_type, results):
        if not self.settings.save_results_to_txt:
            return  # Skip saving if the option is disabled

        results_file = os.path.join(self.output_folder, f"{analysis_type.lower().replace(' ', '_')}_results.txt")
        with open(results_file, "w", encoding='utf-8') as f:
            f.write(f"Results for {analysis_type}:\n")
            if isinstance(results, dict):
                for key, value in results.items():
                    if key != 'image_paths':
                        f.write(f"{key}: {value}\n")
            else:
                f.write(str(results))
        print(success(f"Results saved as txt file: {results_file}"))

    def interpret_results(self, analysis_type, results, table_name):
        technique_info = get_technique_info(analysis_type)

        if isinstance(results, dict) and "Numeric Statistics" in results:
            numeric_stats = results["Numeric Statistics"]
            categorical_stats = results["Categorical Statistics"]
            
            numeric_table = "| Statistic | " + " | ".join(numeric_stats.keys()) + " |\n"
            numeric_table += "| --- | " + " | ".join(["---" for _ in numeric_stats.keys()]) + " |\n"
            for stat in numeric_stats[list(numeric_stats.keys())[0]].keys():
                numeric_table += f"| {stat} | " + " | ".join([f"{numeric_stats[col][stat]:.2f}" for col in numeric_stats.keys()]) + " |\n"
            
            categorical_summary = "\n".join([f"{col}:\n" + "\n".join([f"  - {value}: {count}" for value, count in stats.items()]) for col, stats in categorical_stats.items()])
            
            results_str = f"Numeric Statistics:\n{numeric_table}\n\nCategorical Statistics:\n{categorical_summary}"
        elif isinstance(results, pd.DataFrame):
            results_str = f"DataFrame with shape {results.shape}:\n{results.to_string()}"
        elif isinstance(results, dict):
            results_str = "\n".join([f"{k}: {v}" for k, v in results.items() if k != 'image_paths'])
        else:
            results_str = str(results)

        # Add information about number of visualizations
        num_visualizations = len(results.get('image_paths', []))
        results_str += f"\n\nNumber of visualizations created: {num_visualizations}"

        # Save the results
        self.save_results(analysis_type, results)

        common_prompt = f"""
        Analysis type: {analysis_type}
        Table name: {table_name}
        Database description: {self.database_description}

        Technique Context:
        {technique_info['context']}

        Results:
        {results_str}

        Interpretation Guidelines:
        {technique_info['guidelines']}
        """

        worker_prompt = f"""
        You are an expert data analyst providing insights on exploratory data analysis results. Your task is to interpret the following analysis results and provide a detailed, data-driven interpretation, focusing on discovering patterns and hidden insights. Avoid jargon.

        {common_prompt}

        Please provide a thorough interpretation of these results, highlighting noteworthy patterns, anomalies, or insights. Focus on aspects that would be valuable for business decisions and operational improvements. Always provide specific numbers and percentages.

        Use actual names and values from the data instead of generic references. For example, say "Team George" instead of "Team Alpha", "Product XYZ" instead of "Product Category 1", etc.

        Structure your response in the following format:

        1. Analysis performed and Key Insights:
        [Briefly describe the analysis performed. List at least 2-3 important insights discovered, with relevant numbers and percentages. Provide detailed explanations for each insight.]

        2. Patterns and Trends:
        [Describe at least 2-3 significant patterns or trends observed in the data. Explain their potential significance.]

        3. Potential Issues:
        [Highlight any anomalies, unusual trends, or areas of concern. Mention at least 2-3 potential problems, red flags, audit findings, fraud cases always including relevant numbers and percentages.]

        Ensure your interpretation is comprehensive and focused on actionable insights. While you can be detailed, strive for clarity in your explanations. Use technical terms when necessary, but provide brief explanations for complex concepts.

        Interpretation:
        """

        worker_interpretation = self.worker_erag_api.chat([{"role": "system", "content": "You are an expert data analyst providing insights for business leaders and analysts. Respond in the requested format."}, 
                                                    {"role": "user", "content": worker_prompt}])

        supervisor_prompt = f"""
        You are an expert data analyst providing insights on exploratory data analysis results. Your task is to interpret the following analysis results and provide a detailed, data-driven interpretation.

        {common_prompt}

        Please provide a thorough interpretation of these results, highlighting noteworthy patterns, anomalies, or insights. Focus on the most important aspects that would be valuable for business operations and decision-making. Always provide specific numbers and percentages when discussing findings.
        
        Use actual names and values from the data instead of generic references. For example, use "Team George" instead of "Team Alpha", "Product XYZ" instead of "Product Category 1", etc.
        
        If some data appears to be missing or incomplete, work with the available information without mentioning the limitations. Your goal is to extract as much insight as possible from the given data.
        
        Structure your response in the following format:
        1. Analysis:
        [Provide a detailed description of the analysis performed, including specific metrics and their values]
        2. Key Findings:
        [List the most important discoveries, always including relevant numbers and percentages]
        3. Implications:
        [Discuss the potential impact of these findings on business operations and decision-making]
        4. Operational Recommendations:
        [Suggest concrete operational steps or changes based on these results. Focus on actionable recommendations that can improve business processes, efficiency, or outcomes. Avoid recommending further data analysis.]
        
        Ensure your interpretation is concise yet comprehensive, focusing on actionable insights derived from the data that can be directly applied to business operations.

        Business Analysis:
        """

        supervisor_analysis = self.supervisor_erag_api.chat([
            {"role": "system", "content": "You are a senior business analyst providing insights based on data analysis results. Provide a concise yet comprehensive business analysis."},
            {"role": "user", "content": supervisor_prompt}
        ])

        combined_interpretation = f"""
        Data Analysis:
        {worker_interpretation.strip()}

        Business Analysis:
        {supervisor_analysis.strip()}
        """

        print(success(f"Combined Interpretation for {analysis_type}:"))
        print(combined_interpretation.strip())

        self.text_output += f"\n{combined_interpretation.strip()}\n\n"

        # Save individual interpretation to file
        interpretation_file = os.path.join(self.output_folder, f"{analysis_type.lower().replace(' ', '_')}_interpretation.txt")
        with open(interpretation_file, "w", encoding='utf-8') as f:
            f.write(combined_interpretation.strip())
        print(success(f"Interpretation saved to file: {interpretation_file}"))

        # Handle images for the PDF report
        image_data = []
        if isinstance(results, dict) and 'image_paths' in results:
            for img in results['image_paths']:
                if isinstance(img, tuple) and len(img) == 2:
                    image_data.append(img)
                elif isinstance(img, str):
                    image_data.append((analysis_type, img))

        # Prepare content for PDF report
        pdf_content = f"""
        # {analysis_type}

        ## Data Analysis
        {worker_interpretation.strip()}

        
        ## Business Analysis
        {supervisor_analysis.strip()}
        """

        self.pdf_content.append((analysis_type, image_data, pdf_content))

        # Extract important findings
        self.findings.append(f"{analysis_type}:")
        lines = combined_interpretation.strip().split('\n')
        for i, line in enumerate(lines):
            if line.startswith("1. Analysis performed and Key Insights:") or line.startswith("2. Key Findings:"):
                for finding in lines[i+1:]:
                    if finding.strip() and not finding.startswith(("2.", "3.", "4.")):
                        self.findings.append(finding.strip())
                    elif finding.startswith(("2.", "3.", "4.")):
                        break

        # Update self.image_data for the PDF report
        self.image_data.extend(image_data)

    def save_text_output(self):
        output_file = os.path.join(self.output_folder, "xda_results.txt")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(self.text_output)

    def generate_pdf_report(self):
        report_title = f"Exploratory Data Analysis Report for {self.table_name}"
        
        # Ensure all image data is in the correct format
        formatted_image_data = []
        for item in self.pdf_content:
            analysis_type, images, interpretation = item
            if isinstance(images, list):
                for image in images:
                    if isinstance(image, tuple) and len(image) == 2:
                        formatted_image_data.append(image)
                    elif isinstance(image, str):
                        # If it's just a string (path), use the analysis type as the title
                        formatted_image_data.append((analysis_type, image))
            elif isinstance(images, str):
                # If it's just a string (path), use the analysis type as the title
                formatted_image_data.append((analysis_type, images))
        
        pdf_file = self.pdf_generator.create_enhanced_pdf_report(
            self.findings,
            self.pdf_content,
            formatted_image_data,  # Use the formatted image data
            filename=f"xda_{self.table_name}_report",
            report_title=report_title
        )
        if pdf_file:
            print(success(f"PDF report generated successfully: {pdf_file}"))
            return pdf_file
        else:
            print(error("Failed to generate PDF report"))
            return None