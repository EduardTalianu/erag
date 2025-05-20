# Standard library imports
import os
import logging
import threading
import time
import signal
import sys
from functools import wraps

# Third-party imports
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.tools import diff
from scipy.stats import t, chi2, norm, jarque_bera
from statsmodels.stats.diagnostic import lilliefors
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


class AdvancedExploratoryDataAnalysisB5:
    def __init__(self, worker_erag_api, supervisor_erag_api, db_path):
        self.worker_erag_api = worker_erag_api
        self.supervisor_erag_api = supervisor_erag_api
        self.db_path = db_path
        self.technique_counter = 0
        self.total_techniques = 15
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
        self.entity_names_mapping = {}

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

    def prompt_for_database_description(self):
        """Ask the user for a description of the database"""
        print(info("Please provide a description of the database. This will help the AI models provide better insights."))
        print(info("Describe the purpose, main tables, key data points, and any other relevant information:"))
        self.database_description = input("> ")
        print(success(f"Database description received: {self.database_description}"))

    def identify_entity_names(self, df):
        """
        Identify actual names of entities in the dataset instead of using generic names
        """
        entity_names = {}
        
        # Try to identify teams, customers, products, clusters, etc.
        for col in df.columns:
            if col.lower() in ['team', 'team_name', 'group', 'department']:
                entity_names['teams'] = df[col].unique().tolist()
            elif col.lower() in ['customer', 'client', 'customer_name', 'client_name']:
                entity_names['customers'] = df[col].unique().tolist()
            elif col.lower() in ['product', 'item', 'product_name', 'service']:
                entity_names['products'] = df[col].unique().tolist()
            elif col.lower() in ['cluster', 'segment', 'category']:
                entity_names['clusters'] = df[col].unique().tolist()
            elif col.lower() in ['region', 'location', 'area', 'country', 'city']:
                entity_names['regions'] = df[col].unique().tolist()
        
        # For categorical columns with few unique values, they might be important entities
        for col in df.select_dtypes(include=['object', 'category']).columns:
            unique_values = df[col].unique()
            if 2 <= len(unique_values) <= 20 and col not in entity_names:
                entity_names[col.lower()] = unique_values.tolist()
        
        return entity_names

    @timeout(10)
    def generate_plot(self, plot_function, *args, **kwargs):
        return plot_function(*args, **kwargs)

    def get_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            return [table[0] for table in cursor.fetchall()]

    def run(self):
        print(info(f"Starting Advanced Exploratory Data Analysis (Batch5) on {self.db_path}"))
        
        # Ask for database description before starting analysis
        self.prompt_for_database_description()
        
        tables = self.get_tables()
        for table in tables:
            self.analyze_table(table)
        
        self.save_text_output()
        self.generate_pdf_report()
        print(success(f"Advanced Exploratory Data Analysis (Batch5) completed. Results saved in {self.output_folder}"))

    def analyze_table(self, table_name):
        self.table_name = table_name
        self.output_folder = os.path.join(settings.output_folder, f"axda_b5_{self.table_name}")
        os.makedirs(self.output_folder, exist_ok=True)
        
        self.pdf_generator = PDFReportGenerator(self.output_folder, self.llm_name, self.table_name)
        
        print(highlight(f"\nAnalyzing table: {table_name}"))
        self.text_output += f"\nAnalyzing table: {table_name}\n"
        
        # Load the entire dataset
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            print(info(f"Loaded full dataset with {len(df)} rows and {len(df.columns)} columns"))
        
        # Identify entity names in the dataset
        self.entity_names_mapping = self.identify_entity_names(df)
        entity_description = self.format_entity_description()
        
        print(info(f"Identified entities in the data: {entity_description}"))
        
        analysis_methods = [
            self.cooks_distance_analysis,
            self.stl_decomposition_analysis,
            self.hampel_filter_analysis,
            self.gesd_test_analysis,
            self.dixons_q_test_analysis,
            self.peirce_criterion_analysis,
            self.thompson_tau_test_analysis,
            self.control_charts_analysis,
            self.kde_anomaly_detection_analysis,
            self.hotellings_t_squared_analysis,
            self.breakdown_point_analysis,
            self.chi_square_test_analysis,
            self.simple_thresholding_analysis,
            self.lilliefors_test_analysis,
            self.jarque_bera_test_analysis
        ]

        for method in analysis_methods:
            # Check if script is paused
            self.check_if_paused()
                
            try:
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
            finally:
                self.technique_counter += 1

    def format_entity_description(self):
        """Format the entity names for inclusion in prompts"""
        if not self.entity_names_mapping:
            return "No specific entities identified."
        
        description = []
        for entity_type, names in self.entity_names_mapping.items():
            if names:
                description.append(f"{entity_type.capitalize()}: {', '.join(str(name) for name in names)}")
        
        return "\n".join(description)

    def cooks_distance_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Cook's Distance Analysis"))
        image_paths = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            print(warning("Not enough numeric columns for Cook's Distance analysis."))
            return
        
        X = df[numeric_cols].drop(numeric_cols[-1], axis=1)
        y = df[numeric_cols[-1]]
        
        try:
            # Fit the model
            model = LinearRegression()
            model.fit(X, y)
            
            # Calculate Cook's Distance
            n = len(X)
            p = len(X.columns)
            
            # Calculate MSE
            y_pred = model.predict(X)
            mse = np.mean((y - y_pred) ** 2)
            
            # Calculate leverage
            hat_matrix_diag = X.dot(np.linalg.inv(X.T.dot(X))).dot(X.T).diagonal()
            
            # Calculate Cook's Distance
            residuals = y - y_pred
            cooks_d = (residuals ** 2 / (p * mse)) * (hat_matrix_diag / (1 - hat_matrix_diag) ** 2)
            
            def plot_cooks_distance():
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.calculate_figure_size())
                
                # Stem plot
                ax1.stem(range(len(cooks_d)), cooks_d, markerfmt=",")
                ax1.set_title("Cook's Distance")
                ax1.set_xlabel("Observation")
                ax1.set_ylabel("Cook's Distance")
                
                # Add a threshold line
                threshold = 4 / (n - p)
                ax1.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.4f})')
                ax1.legend()
                
                # Pie chart for influential vs non-influential points
                influential_points = np.sum(cooks_d > threshold)
                non_influential_points = n - influential_points
                ax2.pie([influential_points, non_influential_points], 
                        labels=['Influential', 'Non-influential'], 
                        autopct='%1.1f%%', 
                        startangle=90)
                ax2.set_title('Proportion of Influential Points')
                
                plt.tight_layout()
                return fig, (ax1, ax2)

            result = self.generate_plot(plot_cooks_distance)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_cooks_distance.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
                
                threshold = 4 / (n - p)
                influential_points = np.where(cooks_d > threshold)[0]
                
                # Identify actual entities associated with influential points if possible
                influential_entities = self.identify_influential_entities(df, influential_points)
                
                results = {
                    'image_paths': image_paths,
                    'influential_points': influential_points.tolist(),
                    'influential_entities': influential_entities,
                    'threshold': threshold,
                    'max_cooks_distance': np.max(cooks_d),
                    'mean_cooks_distance': np.mean(cooks_d),
                    'num_influential_points': len(influential_points),
                    'percent_influential': (len(influential_points) / n) * 100
                }
                
                self.interpret_results("Cook's Distance Analysis", results, table_name)
            else:
                print("Skipping Cook's Distance plot due to timeout.")
        
        except Exception as e:
            error_message = f"An error occurred during Cook's Distance analysis: {str(e)}"
            print(error(error_message))
            self.interpret_results("Cook's Distance Analysis", {'error': error_message}, table_name)

    def identify_influential_entities(self, df, influential_indices):
        """Identify actual entities associated with influential points"""
        entities = {}
        
        # Try to find entity columns in the DataFrame
        entity_cols = []
        for col in df.columns:
            if col.lower() in ['team', 'team_name', 'customer', 'product', 'region', 'cluster', 
                             'group', 'department', 'client', 'segment', 'category']:
                entity_cols.append(col)
        
        if not entity_cols:
            # If no obvious entity columns, check categorical columns with few unique values
            for col in df.select_dtypes(include=['object', 'category']).columns:
                if df[col].nunique() <= 20:
                    entity_cols.append(col)
        
        # Get the values for the influential points
        for col in entity_cols:
            influential_values = df.iloc[influential_indices][col].value_counts().to_dict()
            if influential_values:
                entities[col] = influential_values
        
        return entities

    def stl_decomposition_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - STL Decomposition Analysis"))
        image_paths = []
        
        date_cols = df.select_dtypes(include=['datetime64']).columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Try to convert string columns to datetime if no datetime columns are found
        if len(date_cols) == 0:
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    date_cols = [col]
                    print(info(f"Converted column {col} to datetime for STL analysis"))
                    break
                except:
                    continue
        
        if len(date_cols) == 0 or len(numeric_cols) == 0:
            print(warning("No suitable columns for STL decomposition analysis."))
            return
        
        date_col = date_cols[0]
        numeric_col = numeric_cols[0]
        
        df = df.sort_values(by=date_col)
        df = df.set_index(date_col)
        
        # Determine the period based on data frequency
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.inferred_freq == 'D' or df.index.inferred_freq == 'B':
                period = 7  # Weekly seasonality for daily data
            elif df.index.inferred_freq in ['M', 'MS', 'BM', 'BMS']:
                period = 12  # Yearly seasonality for monthly data
            elif df.index.inferred_freq in ['Q', 'QS', 'BQ', 'BQS']:
                period = 4  # Yearly seasonality for quarterly data
            else:
                period = 12  # Default
        else:
            period = 12  # Default
            
        stl = STL(df[numeric_col], period=period)
        result = stl.fit()
        
        def plot_stl_decomposition():
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(self.calculate_figure_size()[0], self.calculate_figure_size()[1]*2))
            ax1.plot(df.index, result.observed)
            ax1.set_ylabel('Observed')
            ax2.plot(df.index, result.trend)
            ax2.set_ylabel('Trend')
            ax3.plot(df.index, result.seasonal)
            ax3.set_ylabel('Seasonal')
            ax4.plot(df.index, result.resid)
            ax4.set_ylabel('Residual')
            plt.tight_layout()
            return fig, (ax1, ax2, ax3, ax4)

        result_plot = self.generate_plot(plot_stl_decomposition)
        if result_plot is not None:
            fig, _ = result_plot
            img_path = os.path.join(self.output_folder, f"{table_name}_stl_decomposition.png")
            plt.savefig(img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            image_paths.append(img_path)
            
            # Get key time periods with significant seasonal patterns
            seasonal_component = pd.Series(result.seasonal, index=df.index)
            key_periods = seasonal_component.groupby(seasonal_component.index.month).mean().sort_values(ascending=False)
            
            results = {
                'image_paths': image_paths,
                'trend_strength': 1 - np.var(result.resid) / np.var(result.trend + result.resid),
                'seasonal_strength': 1 - np.var(result.resid) / np.var(result.seasonal + result.resid),
                'key_seasonal_periods': key_periods.to_dict(),
                'period_used': period,
                'time_column': date_col,
                'value_column': numeric_col
            }
            
            self.interpret_results("STL Decomposition Analysis", results, table_name)
        else:
            print("Skipping STL Decomposition plot due to timeout.")

    def hampel_filter_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Hampel Filter Analysis"))
        image_paths = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            print(warning("No numeric columns for Hampel Filter analysis."))
            return
        
        results = {}
        outlier_entities = {}
        
        for col in numeric_cols:
            series = df[col]
            rolling_median = series.rolling(window=5, center=True).median()
            mad = np.abs(series - rolling_median).rolling(window=5, center=True).median()
            threshold = 3 * 1.4826 * mad
            outliers = np.abs(series - rolling_median) > threshold
            
            # Find actual entities associated with outliers
            if outliers.sum() > 0:
                outlier_indices = np.where(outliers)[0]
                outlier_entities[col] = self.identify_influential_entities(df, outlier_indices)
            
            def plot_hampel():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                ax.plot(series.index, series, label='Original')
                ax.scatter(series.index[outliers], series[outliers], color='red', label='Outliers')
                ax.set_title(f'Hampel Filter Outliers - {col}')
                ax.set_xlabel('Index')
                ax.set_ylabel('Value')
                ax.legend()
                return fig, ax

            result = self.generate_plot(plot_hampel)
            if result is not None:
                fig, ax = result
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_hampel_filter.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
                
                results[col] = {
                    'outliers_count': outliers.sum(),
                    'outliers_percentage': (outliers.sum() / len(series)) * 100,
                    'outlier_values': series[outliers].tolist() if outliers.sum() <= 20 else f"{outliers.sum()} outliers found"
                }
            else:
                print(f"Skipping Hampel Filter plot for {col} due to timeout.")
        
        results['image_paths'] = image_paths
        results['outlier_entities'] = outlier_entities
        
        self.interpret_results("Hampel Filter Analysis", results, table_name)

    def gesd_test_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - GESD Test Analysis"))
        image_paths = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        results = {}
        outlier_entities = {}
        
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) < 3:  # GESD test requires at least 3 data points
                continue
            
            def gesd_test(data, alpha=0.05, max_outliers=10):
                n = len(data)
                if n <= 2:
                    return []
                
                outliers = []
                data_array = data.values.copy()
                indices = np.arange(len(data))
                
                for i in range(max_outliers):
                    if len(data_array) <= 2:
                        break
                    mean = np.mean(data_array)
                    std = np.std(data_array, ddof=1)
                    R = np.max(np.abs(data_array - mean)) / std
                    idx = np.argmax(np.abs(data_array - mean))
                    
                    t_ppf = t.ppf(1 - alpha / (2 * len(data_array)), len(data_array) - 2)
                    lambda_crit = ((len(data_array) - 1) * t_ppf) / np.sqrt((len(data_array) - 2 + t_ppf**2) * len(data_array))
                    
                    if R > lambda_crit:
                        outliers.append((indices[idx], data_array[idx]))
                        data_array = np.delete(data_array, idx)
                        indices = np.delete(indices, idx)
                    else:
                        break
                
                return outliers
            
            outliers = gesd_test(data)
            results[col] = outliers
            
            # Find actual entities associated with outliers
            if outliers:
                outlier_indices = [idx for idx, _ in outliers]
                outlier_entities[col] = self.identify_influential_entities(df, outlier_indices)

            def plot_gesd():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                ax.plot(df.index, df[col], label='Original Data')
                outlier_indices = [idx for idx, _ in outliers]
                outlier_values = [val for _, val in outliers]
                ax.scatter(df.index[outlier_indices], outlier_values, color='red', label='Outliers')
                ax.set_title(f'GESD Test Outliers - {col}')
                ax.set_xlabel('Index')
                ax.set_ylabel('Value')
                ax.legend()
                return fig, ax

            result = self.generate_plot(plot_gesd)
            if result is not None:
                fig, ax = result
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_gesd_test.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
            else:
                print(f"Skipping GESD Test plot for {col} due to timeout.")
        
        results['image_paths'] = image_paths
        results['outlier_entities'] = outlier_entities
        
        self.interpret_results("GESD Test Analysis", results, table_name)

    def dixons_q_test_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Dixon's Q Test Analysis"))
        image_paths = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        results = {}
        outlier_entities = {}
        
        for col in numeric_cols:
            data = df[col].dropna().sort_values()
            if len(data) < 3 or len(data) > 30:  # Dixon's Q test is typically used for sample sizes between 3 and 30
                continue
            
            def dixon_q_test(data, alpha=0.05):
                n = len(data)
                if n < 3 or n > 30:
                    return None, None, [], []
                
                q_crit_table = {
                    3: 0.970, 4: 0.829, 5: 0.710, 6: 0.628, 7: 0.569, 8: 0.608, 9: 0.564, 10: 0.530,
                    11: 0.502, 12: 0.479, 13: 0.611, 14: 0.586, 15: 0.565, 16: 0.546, 17: 0.529,
                    18: 0.514, 19: 0.501, 20: 0.489, 21: 0.478, 22: 0.468, 23: 0.459, 24: 0.451,
                    25: 0.443, 26: 0.436, 27: 0.429, 28: 0.423, 29: 0.417, 30: 0.412
                }
                
                q_crit = q_crit_table[n]
                
                if n <= 7:
                    q_low = (data[1] - data[0]) / (data[-1] - data[0])
                    q_high = (data[-1] - data[-2]) / (data[-1] - data[0])
                else:
                    q_low = (data[1] - data[0]) / (data[-2] - data[0])
                    q_high = (data[-1] - data[-2]) / (data[-1] - data[1])
                
                outlier_low = q_low > q_crit
                outlier_high = q_high > q_crit
                
                # Get indices of outliers
                low_indices = [data.index[0]] if outlier_low else []
                high_indices = [data.index[-1]] if outlier_high else []
                
                return (data[0] if outlier_low else None, 
                        data[-1] if outlier_high else None,
                        low_indices,
                        high_indices)
            
            low_outlier, high_outlier, low_indices, high_indices = dixon_q_test(data)
            results[col] = {'low_outlier': low_outlier, 'high_outlier': high_outlier}
            
            # Find actual entities associated with outliers
            all_outlier_indices = low_indices + high_indices
            if all_outlier_indices:
                outlier_entities[col] = self.identify_influential_entities(df, all_outlier_indices)

            def plot_dixon_q():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                ax.plot(range(len(data)), data, 'bo-')
                if low_outlier is not None:
                    ax.plot(0, low_outlier, 'ro', markersize=10, label='Low Outlier')
                if high_outlier is not None:
                    ax.plot(len(data)-1, high_outlier, 'go', markersize=10, label='High Outlier')
                ax.set_title(f"Dixon's Q Test - {col}")
                ax.set_xlabel('Sorted Data Index')
                ax.set_ylabel('Value')
                ax.legend()
                return fig, ax

            result = self.generate_plot(plot_dixon_q)
            if result is not None:
                fig, ax = result
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_dixon_q_test.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
            else:
                print(f"Skipping Dixon's Q Test plot for {col} due to timeout.")
        
        results['image_paths'] = image_paths
        results['outlier_entities'] = outlier_entities
        
        self.interpret_results("Dixon's Q Test Analysis", results, table_name)

    def peirce_criterion_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Peirce's Criterion Analysis"))
        image_paths = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        results = {}
        outlier_entities = {}
        
        def peirce_criterion(data):
            n = len(data)
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            
            # Peirce's criterion table (approximation)
            R_table = {1: 1.0, 2: 1.28, 3: 1.38, 4: 1.44, 5: 1.48, 6: 1.51, 7: 1.53, 8: 1.55, 9: 1.57, 10: 1.58}
            
            if n <= 10:
                R = R_table[n]
            else:
                R = 1.58 + 0.2 * np.log10(n / 10)
            
            threshold = R * std
            outliers = np.where(abs(data - mean) > threshold)[0]
            outlier_values = data[outliers]
            
            return outliers, outlier_values
        
        for col in numeric_cols:
            data = df[col].dropna().values
            outlier_indices, outlier_values = peirce_criterion(data)
            results[col] = outlier_values.tolist()
            
            # Find actual entities associated with outliers
            if len(outlier_indices) > 0:
                outlier_entities[col] = self.identify_influential_entities(df, outlier_indices)

            def plot_peirce():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                ax.plot(np.arange(len(data)), data, label='Original Data')
                ax.scatter(outlier_indices, outlier_values, color='red', label='Outliers')
                ax.set_title(f"Peirce's Criterion Outliers - {col}")
                ax.set_xlabel('Index')
                ax.set_ylabel('Value')
                ax.legend()
                return fig, ax

            result = self.generate_plot(plot_peirce)
            if result is not None:
                fig, ax = result
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_peirce_criterion.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
            else:
                print(f"Skipping Peirce's Criterion plot for {col} due to timeout.")
        
        results['image_paths'] = image_paths
        results['outlier_entities'] = outlier_entities
        
        self.interpret_results("Peirce's Criterion Analysis", results, table_name)

    def thompson_tau_test_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Thompson Tau Test Analysis"))
        image_paths = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        results = {}
        outlier_entities = {}
        
        def thompson_tau_test(data, alpha=0.05):
            n = len(data)
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            
            t_value = t.ppf(1 - alpha / 2, n - 2)
            tau = (t_value * (n - 1)) / (np.sqrt(n) * np.sqrt(n - 2 + t_value**2))
            
            delta = tau * std
            outlier_indices = np.where(abs(data - mean) > delta)[0]
            outlier_values = data[outlier_indices]
            
            return outlier_indices, outlier_values
        
        for col in numeric_cols:
            data = df[col].dropna().values
            outlier_indices, outlier_values = thompson_tau_test(data)
            results[col] = outlier_values.tolist()
            
            # Find actual entities associated with outliers
            if len(outlier_indices) > 0:
                outlier_entities[col] = self.identify_influential_entities(df, outlier_indices)

            def plot_thompson_tau():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                ax.plot(np.arange(len(data)), data, label='Original Data')
                ax.scatter(outlier_indices, outlier_values, color='red', label='Outliers')
                ax.set_title(f'Thompson Tau Test Outliers - {col}')
                ax.set_xlabel('Index')
                ax.set_ylabel('Value')
                ax.legend()
                return fig, ax

            result = self.generate_plot(plot_thompson_tau)
            if result is not None:
                fig, ax = result
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_thompson_tau.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
            else:
                print(f"Skipping Thompson Tau Test plot for {col} due to timeout.")
        
        results['image_paths'] = image_paths
        results['outlier_entities'] = outlier_entities
        
        self.interpret_results("Thompson Tau Test Analysis", results, table_name)

    def control_charts_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Control Charts Analysis (CUSUM, EWMA)"))
        image_paths = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        results = {}
        out_of_control_entities = {}
        
        def cusum_chart(data, threshold=1, drift=0):
            cumsum = np.zeros(len(data))
            for i in range(1, len(data)):
                cumsum[i] = max(0, cumsum[i-1] + data[i] - (np.mean(data) + drift))
            
            upper_control_limit = threshold * np.std(data)
            return cumsum, upper_control_limit
        
        def ewma_chart(data, lambda_param=0.2, L=3):
            ewma = np.zeros(len(data))
            ewma[0] = data[0]
            for i in range(1, len(data)):
                ewma[i] = lambda_param * data[i] + (1 - lambda_param) * ewma[i-1]
            
            std_ewma = np.std(data) * np.sqrt(lambda_param / (2 - lambda_param))
            upper_control_limit = np.mean(data) + L * std_ewma
            lower_control_limit = np.mean(data) - L * std_ewma
            
            return ewma, upper_control_limit, lower_control_limit
        
        for col in numeric_cols:
            data = df[col].dropna().values
            
            cusum, cusum_ucl = cusum_chart(data)
            ewma, ewma_ucl, ewma_lcl = ewma_chart(data)
            
            # Identify out-of-control points
            cusum_out_indices = np.where(cusum > cusum_ucl)[0]
            ewma_out_indices = np.where((ewma > ewma_ucl) | (ewma < ewma_lcl))[0]
            
            # Combine all out-of-control indices
            all_out_indices = np.unique(np.concatenate([cusum_out_indices, ewma_out_indices]) if len(cusum_out_indices) > 0 and len(ewma_out_indices) > 0 else 
                                        (cusum_out_indices if len(cusum_out_indices) > 0 else ewma_out_indices))
            
            # Find actual entities associated with out-of-control points
            if len(all_out_indices) > 0:
                out_of_control_entities[col] = self.identify_influential_entities(df, all_out_indices)
            
            def plot_control_charts():
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=self.calculate_figure_size())
                
                # CUSUM chart
                ax1.plot(cusum, label='CUSUM')
                ax1.axhline(y=cusum_ucl, color='r', linestyle='--', label='Upper Control Limit')
                ax1.set_title(f'CUSUM Control Chart for {col}')
                ax1.set_xlabel('Observation')
                ax1.set_ylabel('Cumulative Sum')
                ax1.legend()
                
                # EWMA chart
                ax2.plot(ewma, label='EWMA')
                ax2.axhline(y=ewma_ucl, color='r', linestyle='--', label='Upper Control Limit')
                ax2.axhline(y=ewma_lcl, color='r', linestyle='--', label='Lower Control Limit')
                ax2.set_title(f'EWMA Control Chart for {col}')
                ax2.set_xlabel('Observation')
                ax2.set_ylabel('EWMA')
                ax2.legend()
                
                # Pie chart for out-of-control points
                cusum_out = np.sum(cusum > cusum_ucl)
                ewma_out = np.sum((ewma > ewma_ucl) | (ewma < ewma_lcl))
                total_points = len(data)
                in_control = total_points - cusum_out - ewma_out
                
                ax3.pie([in_control, cusum_out, ewma_out], 
                        labels=['In Control', 'CUSUM Out', 'EWMA Out'],
                        autopct='%1.1f%%',
                        startangle=90)
                ax3.set_title('Distribution of Control Points')
                
                plt.tight_layout()
                return fig, (ax1, ax2, ax3)
            
            result = self.generate_plot(plot_control_charts)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_control_charts.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
                
                results[col] = {
                    'cusum_out_of_control': np.sum(cusum > cusum_ucl),
                    'ewma_out_of_control': np.sum((ewma > ewma_ucl) | (ewma < ewma_lcl)),
                    'total_out_of_control': len(all_out_indices),
                    'percentage_out_of_control': (len(all_out_indices) / len(data)) * 100
                }
            else:
                print(f"Skipping control charts for {col} due to timeout.")
        
        results['image_paths'] = image_paths
        results['out_of_control_entities'] = out_of_control_entities
        
        self.interpret_results("Control Charts Analysis (CUSUM, EWMA)", results, table_name)

    def kde_anomaly_detection_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - KDE Anomaly Detection Analysis"))
        image_paths = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        results = {}
        anomaly_entities = {}
        
        for col in numeric_cols:
            data = df[col].dropna().values
            
            kde = stats.gaussian_kde(data)
            x_range = np.linspace(min(data), max(data), 1000)
            density = kde(x_range)
            
            threshold = np.percentile(kde(data), 5)  # Use 5th percentile as anomaly threshold
            anomaly_indices = np.where(kde(data) < threshold)[0]
            anomalies = data[anomaly_indices]
            
            # Find actual entities associated with anomalies
            if len(anomaly_indices) > 0:
                anomaly_entities[col] = self.identify_influential_entities(df, anomaly_indices)
            
            def plot_kde_anomalies():
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.calculate_figure_size())
                ax1.plot(x_range, density, label='KDE')
                ax1.axhline(y=threshold, color='r', linestyle='--', label='Anomaly Threshold')
                ax1.scatter(anomalies, kde(anomalies), color='r', label='Anomalies')
                ax1.set_title(f'KDE Anomaly Detection for {col}')
                ax1.set_xlabel('Value')
                ax1.set_ylabel('Density')
                ax1.legend()

                # Pie chart
                normal_count = len(data) - len(anomalies)
                ax2.pie([normal_count, len(anomalies)], 
                        labels=['Normal', 'Anomalies'],
                        autopct='%1.1f%%',
                        startangle=90)
                ax2.set_title('Distribution of Anomalies')

                plt.tight_layout()
                return fig, (ax1, ax2)

            result = self.generate_plot(plot_kde_anomalies)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_kde_anomalies.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
                
                results[col] = {
                    'anomalies_count': len(anomalies),
                    'anomalies_percentage': (len(anomalies) / len(data)) * 100,
                    'anomaly_values': anomalies.tolist() if len(anomalies) <= 20 else f"{len(anomalies)} anomalies found"
                }
            else:
                print(f"Skipping KDE anomaly detection plot for {col} due to timeout.")
        
        results['image_paths'] = image_paths
        results['anomaly_entities'] = anomaly_entities
        
        self.interpret_results("KDE Anomaly Detection Analysis", results, table_name)

    def hotellings_t_squared_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Hotelling's T-squared Analysis"))
        image_paths = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            print(warning("Not enough numeric columns for Hotelling's T-squared analysis."))
            return
        
        X = df[numeric_cols].dropna()
        n, p = X.shape
        
        if n <= p:
            print(warning("Not enough samples for Hotelling's T-squared analysis."))
            return
        
        mean = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False)
        
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            print(warning("Singular matrix encountered. Using pseudo-inverse instead."))
            inv_cov = np.linalg.pinv(cov)
        
        def t_squared(x):
            diff = x - mean
            return np.dot(np.dot(diff, inv_cov), diff.T)
        
        t_sq = np.array([t_squared(x) for x in X.values])
        
        # Calculate critical value
        f_crit = stats.f.ppf(0.95, p, n-p)
        t_sq_crit = ((n-1)*p/(n-p)) * f_crit
        
        outlier_indices = np.where(t_sq > t_sq_crit)[0]
        outliers = X.iloc[outlier_indices]
        
        # Find actual entities associated with outliers
        outlier_entities = {}
        if len(outlier_indices) > 0:
            outlier_entities = self.identify_influential_entities(df, outlier_indices)
        
        def plot_t_squared():
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.calculate_figure_size())
            ax1.plot(t_sq, label="T-squared")
            ax1.axhline(y=t_sq_crit, color='r', linestyle='--', label='Critical Value')
            ax1.set_title("Hotelling's T-squared Control Chart")
            ax1.set_xlabel('Observation')
            ax1.set_ylabel('T-squared')
            ax1.legend()

            # Pie chart
            normal_count = n - len(outliers)
            ax2.pie([normal_count, len(outliers)], 
                    labels=['Normal', 'Outliers'],
                    autopct='%1.1f%%',
                    startangle=90)
            ax2.set_title('Distribution of Outliers')

            plt.tight_layout()
            return fig, (ax1, ax2)

        result = self.generate_plot(plot_t_squared)
        if result is not None:
            fig, _ = result
            img_path = os.path.join(self.output_folder, f"{table_name}_hotellings_t_squared.png")
            plt.savefig(img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            image_paths.append(img_path)
            
            results = {
                'image_paths': image_paths,
                'outliers_count': len(outliers),
                'outliers_percentage': (len(outliers) / n) * 100,
                'outlier_entities': outlier_entities
            }
            
            self.interpret_results("Hotelling's T-squared Analysis", results, table_name)
        else:
            print("Skipping Hotelling's T-squared plot due to timeout.")

    def breakdown_point_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Breakdown Point Analysis"))
        image_paths = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        results = {}
        
        for col in numeric_cols:
            data = df[col].dropna().values
            n = len(data)
            
            # Calculate breakdown point for mean and median
            bp_mean = 1 / n
            bp_median = 0.5
            
            # Calculate trimmed mean with different trimming levels
            trim_levels = [0.1, 0.2, 0.3]
            trimmed_means = [stats.trim_mean(data, trim) for trim in trim_levels]
            
            results[col] = {
                'bp_mean': bp_mean,
                'bp_median': bp_median,
                'trimmed_means': dict(zip(trim_levels, trimmed_means)),
                'mean': np.mean(data),
                'median': np.median(data),
                'std_dev': np.std(data)
            }

            def plot_breakdown_point():
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.calculate_figure_size())
                ax1.bar(['Mean', 'Median'], [bp_mean, bp_median])
                ax1.set_title(f'Breakdown Point - {col}')
                ax1.set_ylabel('Breakdown Point')
                
                ax2.plot(trim_levels, trimmed_means, marker='o')
                ax2.set_title(f'Trimmed Means - {col}')
                ax2.set_xlabel('Trimming Level')
                ax2.set_ylabel('Trimmed Mean')
                
                plt.tight_layout()
                return fig, (ax1, ax2)

            result = self.generate_plot(plot_breakdown_point)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_breakdown_point.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
            else:
                print(f"Skipping Breakdown Point plot for {col} due to timeout.")
        
        results['image_paths'] = image_paths
        
        self.interpret_results("Breakdown Point Analysis", results, table_name)

    def chi_square_test_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Chi-Square Test Analysis"))
        image_paths = []
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        results = {}
        
        for col in categorical_cols:
            observed = df[col].value_counts()
            n = len(df)
            expected = pd.Series(n/len(observed), index=observed.index)
            
            chi2, p_value = stats.chisquare(observed, expected)
            
            results[col] = {
                'chi2_statistic': chi2,
                'p_value': p_value,
                'degrees_of_freedom': len(observed) - 1,
                'observed_counts': observed.to_dict(),
                'expected_counts': expected.to_dict(),
                'categories': observed.index.tolist()
            }

            def plot_chi_square():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                x = np.arange(len(observed))
                width = 0.35
                ax.bar(x - width/2, observed, width, label='Observed')
                ax.bar(x + width/2, expected, width, label='Expected')
                ax.set_xlabel('Categories')
                ax.set_ylabel('Frequency')
                ax.set_title(f'Chi-Square Test - {col}')
                ax.set_xticks(x)
                ax.set_xticklabels(observed.index, rotation=45, ha='right')
                ax.legend()
                plt.tight_layout()
                return fig, ax

            result = self.generate_plot(plot_chi_square)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_chi_square.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
            else:
                print(f"Skipping Chi-Square Test plot for {col} due to timeout.")
        
        results['image_paths'] = image_paths
        
        self.interpret_results("Chi-Square Test Analysis", results, table_name)

    def simple_thresholding_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Simple Thresholding Analysis"))
        image_paths = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        results = {}
        outlier_entities = {}
        
        for col in numeric_cols:
            data = df[col].dropna()
            
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            outlier_indices = np.where((data < lower_bound) | (data > upper_bound))[0]
            
            # Find actual entities associated with outliers
            if len(outlier_indices) > 0:
                outlier_entities[col] = self.identify_influential_entities(df, outlier_indices)
            
            results[col] = {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outliers_count': len(outliers),
                'outliers_percentage': (len(outliers) / len(data)) * 100,
                'outlier_values': outliers.tolist() if len(outliers) <= 20 else f"{len(outliers)} outliers found"
            }

            def plot_simple_thresholding():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                ax.boxplot(data)
                ax.scatter(np.ones(len(outliers)), outliers, color='red', label='Outliers')
                ax.set_title(f'Simple Thresholding - {col}')
                ax.set_ylabel('Value')
                ax.legend()
                return fig, ax

            result = self.generate_plot(plot_simple_thresholding)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_simple_thresholding.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
            else:
                print(f"Skipping Simple Thresholding plot for {col} due to timeout.")
        
        results['image_paths'] = image_paths
        results['outlier_entities'] = outlier_entities
        
        self.interpret_results("Simple Thresholding Analysis", results, table_name)

    def lilliefors_test_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Lilliefors Test Analysis"))
        image_paths = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        results = {}
        
        for col in numeric_cols:
            data = df[col].dropna().values
            
            statistic, p_value = lilliefors(data)
            
            results[col] = {
                'test_statistic': statistic,
                'p_value': p_value,
                'is_normal': p_value > 0.05,
                'mean': np.mean(data),
                'std_dev': np.std(data),
                'skewness': stats.skew(data),
                'kurtosis': stats.kurtosis(data)
            }

            def plot_lilliefors():
                fig, ax = plt.subplots(figsize=self.calculate_figure_size())
                stats.probplot(data, dist="norm", plot=ax)
                ax.set_title(f'Lilliefors Test Q-Q Plot - {col}')
                return fig, ax

            result = self.generate_plot(plot_lilliefors)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_lilliefors.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
            else:
                print(f"Skipping Lilliefors Test plot for {col} due to timeout.")
        
        results['image_paths'] = image_paths
        
        self.interpret_results("Lilliefors Test Analysis", results, table_name)

    def jarque_bera_test_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Jarque-Bera Test Analysis"))
        image_paths = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        results = {}
        
        for col in numeric_cols:
            data = df[col].dropna().values
            
            # Get just the stat and p-value from jarque_bera
            statistic, p_value = jarque_bera(data)
            
            # Calculate skewness and kurtosis separately
            skew = stats.skew(data)
            kurtosis = stats.kurtosis(data)
            
            results[col] = {
                'test_statistic': statistic,
                'p_value': p_value,
                'skewness': skew,
                'kurtosis': kurtosis,
                'is_normal': p_value > 0.05,
                'mean': np.mean(data),
                'std_dev': np.std(data)
            }

            def plot_jarque_bera():
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.calculate_figure_size())
                sns.histplot(data, kde=True, ax=ax1)
                ax1.set_title(f'Distribution - {col}')
                
                stats.probplot(data, dist="norm", plot=ax2)
                ax2.set_title(f'Q-Q Plot - {col}')
                
                plt.tight_layout()
                return fig, (ax1, ax2)

            result = self.generate_plot(plot_jarque_bera)
            if result is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_{col}_jarque_bera.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(img_path)
            else:
                print(f"Skipping Jarque-Bera Test plot for {col} due to timeout.")
        
        results['image_paths'] = image_paths
        
        self.interpret_results("Jarque-Bera Test Analysis", results, table_name)

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

        # Add entity information
        entity_description = self.format_entity_description()

        common_prompt = f"""
        Analysis type: {analysis_type}
        Table name: {table_name}
        
        Database Description:
        {self.database_description}
        
        Entities in the data:
        {entity_description}

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
        
        When referring to entities in the data (like teams, products, regions, etc.), use their actual names from the data rather than generic terms like "Team A" or "Cluster 1".

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
        
        When referring to entities in the data (like teams, products, regions, etc.), use their actual names from the data rather than generic terms like "Team A" or "Cluster 1".
        
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
        
        # Save raw results to file
        results_file = os.path.join(self.output_folder, f"{analysis_type.lower().replace(' ', '_')}_results.txt")
        with open(results_file, "w", encoding='utf-8') as f:
            f.write(f"Results for {analysis_type}:\n\n")
            f.write(results_str)
        print(success(f"Results saved to file: {results_file}"))

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
        output_file = os.path.join(self.output_folder, "axda_b5_results.txt")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(self.text_output)

    def generate_pdf_report(self):
        report_title = f"Advanced Exploratory Data Analysis (Batch 5) Report for {self.table_name}"
        
        # Add database description to the report
        if self.database_description:
            report_title += f"\nDatabase Description: {self.database_description}"
        
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
            filename=f"axda_b5_{self.table_name}_report",
            report_title=report_title
        )
        if pdf_file:
            print(success(f"PDF report generated successfully: {pdf_file}"))
            return pdf_file
        else:
            print(error("Failed to generate PDF report"))
            return None