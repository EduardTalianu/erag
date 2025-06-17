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

class TimeoutException(Exception):
    pass

class FinancialExploratoryDataAnalysis:
    def __init__(self, worker_erag_api, supervisor_erag_api, db_path):
        self.worker_erag_api = worker_erag_api
        self.supervisor_erag_api = supervisor_erag_api
        self.db_path = db_path
        self.technique_counter = 1
        self.total_techniques = 10
        self.table_name = None
        self.output_folder = None
        self.text_output = ""
        self.pdf_content = []
        self.findings = []
        self.llm_name = f"Worker: {self.worker_erag_api.model}, Supervisor: {self.supervisor_erag_api.model}"
        self.toc_entries = []
        self.executive_summary = ""
        self.image_paths = []
        self.max_pixels = 400000
        self.timeout_seconds = 10
        self.data = {}
        self.all_image_paths = []
        self.image_data = []
        self.paused = False
        self.database_description = ""
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
            time.sleep(0.1)

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
                thread.daemon = True  # Make thread daemon to avoid hanging
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

    def generate_plot_safely(self, plot_function, *args, **kwargs):
        """Safely generate plots with error handling"""
        try:
            return plot_function(*args, **kwargs)
        except Exception as e:
            print(f"Warning: Plot generation failed: {str(e)}")
            return None, None

    def prompt_for_database_description(self):
        """Ask the user for a description of the database"""
        print(info("Please provide a description of the financial database. This will help the AI models provide better insights."))
        print(info("Describe the client portfolio, business context, industry, and any other relevant information:"))
        self.database_description = input("> ")
        print(success(f"Database description received: {self.database_description}"))

    def run(self):
        self.prompt_for_database_description()
        print(info(f"Starting Financial Exploratory Data Analysis on {self.db_path}"))
        tables = self.get_tables()
        for table in tables:
            self.analyze_table(table)
        
        print(info("Generating Executive Summary..."))
        self.generate_executive_summary()
        self.save_text_output()
        self.generate_pdf_report()
        print(success(f"Financial Exploratory Data Analysis completed. Results saved in {self.output_folder}"))

    def get_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'information_schema%';")
            return [table[0] for table in cursor.fetchall()]

    def analyze_table(self, table_name):
        self.table_name = table_name
        self.output_folder = os.path.join(os.path.dirname(self.db_path), "fxda_output")
        os.makedirs(self.output_folder, exist_ok=True)
        
        print(highlight(f"\nAnalyzing financial table: {table_name}"))
        self.text_output += f"\nAnalyzing financial table: {table_name}\n"
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(f"SELECT * FROM '{table_name}'", conn)
            print(info(f"Loaded financial dataset with {len(df)} clients and {len(df.columns)} financial metrics"))

        # Store the data for cross-analysis
        self.data[table_name] = df

        analysis_methods = [
            self.portfolio_overview_analysis,
            self.profitability_analysis,
            self.leverage_and_risk_analysis,
            self.liquidity_analysis,
            self.efficiency_analysis,
            self.client_segmentation_analysis,
            self.risk_assessment_analysis,
            self.correlation_analysis,
            self.outlier_detection_analysis,
            self.benchmarking_analysis
        ]

        for method in analysis_methods:
            try:
                self.check_if_paused()
                method(df, table_name)
            except Exception as e:
                error_message = f"An error occurred during {method.__name__}: {str(e)}"
                print(error(error_message))
                self.text_output += f"\n{error_message}\n"
                self.pdf_content.append((method.__name__, [], error_message))

    def portfolio_overview_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Portfolio Overview Analysis"))
        
        # Get financial metrics columns (excluding ID)
        financial_metrics = [col for col in df.columns if col.upper() in [
            'EBITDA', 'LEVERAGE', 'DSCR', 'TURNOVER', 'OPERATING_OUTPUT', 'TOTAL_ASSETS',
            'EQUITY_CAPITAL', 'PRE_TAX_PROFIT', 'BALANCE_SHEET_TOTAL', 'ROA', 'NET_PROFIT_OR_LOSS_CURRENT_YEAR',
            'NET_SALES_CURRENT_YEAR', 'WORKING_CAPITAL_CURRENT_YEAR', 'EBIT', 'FREE_CASHFLOW'
        ]]
        
        results = {
            "total_clients": len(df),
            "financial_metrics_count": len(financial_metrics),
            "portfolio_metrics": {}
        }
        
        # Calculate portfolio-wide statistics
        for metric in financial_metrics:
            if metric in df.columns:
                metric_data = pd.to_numeric(df[metric], errors='coerce')
                results["portfolio_metrics"][metric] = {
                    'total_sum': float(metric_data.sum()),
                    'average': float(metric_data.mean()),
                    'median': float(metric_data.median()),
                    'std_dev': float(metric_data.std()),
                    'min_value': float(metric_data.min()),
                    'max_value': float(metric_data.max()),
                    'clients_with_data': int(metric_data.notna().sum()),
                    'missing_data_pct': float((metric_data.isna().sum() / len(df)) * 100)
                }

        def plot_portfolio_overview():
            # Select key metrics for visualization
            key_metrics = ['TURNOVER', 'TOTAL_ASSETS', 'EBITDA', 'NET_PROFIT_OR_LOSS_CURRENT_YEAR']
            available_metrics = [m for m in key_metrics if m in df.columns]
            
            if len(available_metrics) >= 2:
                fig, axes = plt.subplots(2, 2, figsize=(20, 16))
                axes = axes.flatten()
                
                for i, metric in enumerate(available_metrics[:4]):
                    if i < 4:
                        metric_data = pd.to_numeric(df[metric], errors='coerce').dropna()
                        
                        # Histogram
                        axes[i].hist(metric_data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                        axes[i].set_title(f'Distribution of {metric}')
                        axes[i].set_xlabel(metric)
                        axes[i].set_ylabel('Number of Clients')
                        
                        # Add statistics text
                        mean_val = metric_data.mean()
                        median_val = metric_data.median()
                        axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
                        axes[i].axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
                        axes[i].legend()
                
                plt.tight_layout()
                return fig, axes
            else:
                return None, None

        result = self.generate_plot_safely(plot_portfolio_overview)
        if result is not None and result[0] is not None:
            fig, _ = result
            img_path = os.path.join(self.output_folder, f"{table_name}_portfolio_overview.png")
            plt.savefig(img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            results['image_paths'] = [("Portfolio Overview", img_path)]
        
        self.interpret_results("Portfolio Overview Analysis", results, table_name)
        self.technique_counter += 1

    def profitability_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Profitability Analysis"))
        
        # Identify profitability columns
        profit_cols = [col for col in df.columns if any(keyword in col.upper() for keyword in [
            'PROFIT', 'EBITDA', 'EBIT', 'ROA', 'NET_SALES', 'TURNOVER'
        ])]
        
        results = {"profitability_metrics": {}}
        image_paths = []
        
        for col in profit_cols:
            if col in df.columns:
                profit_data = pd.to_numeric(df[col], errors='coerce').dropna()
                
                results["profitability_metrics"][col] = {
                    'mean': float(profit_data.mean()),
                    'median': float(profit_data.median()),
                    'std_dev': float(profit_data.std()),
                    'positive_clients': int((profit_data > 0).sum()),
                    'negative_clients': int((profit_data <= 0).sum()),
                    'positive_percentage': float(((profit_data > 0).sum() / len(profit_data)) * 100),
                    'top_10_percentile': float(profit_data.quantile(0.9)),
                    'bottom_10_percentile': float(profit_data.quantile(0.1))
                }

        # Create profitability comparison chart
        def plot_profitability_comparison():
            if len(profit_cols) >= 2:
                available_cols = [col for col in profit_cols if col in df.columns][:4]
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
                
                # Box plot comparison
                box_data = []
                box_labels = []
                for col in available_cols:
                    col_data = pd.to_numeric(df[col], errors='coerce').dropna()
                    box_data.append(col_data)
                    box_labels.append(col)
                
                ax1.boxplot(box_data, labels=box_labels)
                ax1.set_title('Profitability Metrics Comparison (Box Plot)')
                ax1.set_ylabel('Values')
                ax1.tick_params(axis='x', rotation=45)
                
                # Scatter plot if we have ROA and another metric
                if 'ROA' in available_cols and len(available_cols) > 1:
                    other_metric = [col for col in available_cols if col != 'ROA'][0]
                    roa_data = pd.to_numeric(df['ROA'], errors='coerce')
                    other_data = pd.to_numeric(df[other_metric], errors='coerce')
                    
                    ax2.scatter(roa_data, other_data, alpha=0.6)
                    ax2.set_xlabel('ROA')
                    ax2.set_ylabel(other_metric)
                    ax2.set_title(f'ROA vs {other_metric}')
                    
                    # Add correlation coefficient
                    corr_coef = roa_data.corr(other_data)
                    ax2.text(0.05, 0.95, f'Correlation: {corr_coef:.3f}', 
                            transform=ax2.transAxes, verticalalignment='top')
                
                plt.tight_layout()
                return fig, (ax1, ax2)
            return None, None

        result = self.generate_plot_safely(plot_profitability_comparison)
        if result is not None and result[0] is not None:
            fig, _ = result
            img_path = os.path.join(self.output_folder, f"{table_name}_profitability_analysis.png")
            plt.savefig(img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            image_paths.append(("Profitability Analysis", img_path))

        results['image_paths'] = image_paths
        self.interpret_results("Profitability Analysis", results, table_name)
        self.technique_counter += 1

    def leverage_and_risk_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Leverage and Risk Analysis"))
        
        # Identify leverage and risk columns
        risk_cols = [col for col in df.columns if any(keyword in col.upper() for keyword in [
            'LEVERAGE', 'DSCR', 'DEBT', 'INTEREST', 'COVERAGE'
        ])]
        
        results = {"risk_metrics": {}}
        image_paths = []
        
        for col in risk_cols:
            if col in df.columns:
                risk_data = pd.to_numeric(df[col], errors='coerce').dropna()
                
                # Define risk thresholds based on common financial standards
                if 'LEVERAGE' in col.upper():
                    high_risk_threshold = 3.0
                    medium_risk_threshold = 2.0
                elif 'DSCR' in col.upper():
                    high_risk_threshold = 1.0  # Below 1.0 is high risk
                    medium_risk_threshold = 1.25
                else:
                    high_risk_threshold = risk_data.quantile(0.75)
                    medium_risk_threshold = risk_data.quantile(0.5)
                
                if 'DSCR' in col.upper():
                    high_risk_clients = int((risk_data < high_risk_threshold).sum())
                    medium_risk_clients = int(((risk_data >= high_risk_threshold) & (risk_data < medium_risk_threshold)).sum())
                    low_risk_clients = int((risk_data >= medium_risk_threshold).sum())
                else:
                    high_risk_clients = int((risk_data > high_risk_threshold).sum())
                    medium_risk_clients = int(((risk_data <= high_risk_threshold) & (risk_data > medium_risk_threshold)).sum())
                    low_risk_clients = int((risk_data <= medium_risk_threshold).sum())
                
                results["risk_metrics"][col] = {
                    'mean': float(risk_data.mean()),
                    'median': float(risk_data.median()),
                    'std_dev': float(risk_data.std()),
                    'high_risk_clients': high_risk_clients,
                    'medium_risk_clients': medium_risk_clients,
                    'low_risk_clients': low_risk_clients,
                    'high_risk_percentage': float((high_risk_clients / len(risk_data)) * 100),
                    'risk_thresholds': {
                        'high_risk': float(high_risk_threshold),
                        'medium_risk': float(medium_risk_threshold)
                    }
                }

        def plot_risk_analysis():
            if len(risk_cols) >= 1:
                available_cols = [col for col in risk_cols if col in df.columns][:2]
                
                fig, axes = plt.subplots(1, len(available_cols), figsize=(10*len(available_cols), 8))
                if len(available_cols) == 1:
                    axes = [axes]
                
                for i, col in enumerate(available_cols):
                    risk_data = pd.to_numeric(df[col], errors='coerce').dropna()
                    
                    # Risk distribution pie chart
                    if col in results["risk_metrics"]:
                        risk_metrics = results["risk_metrics"][col]
                        sizes = [risk_metrics['high_risk_clients'], 
                                risk_metrics['medium_risk_clients'], 
                                risk_metrics['low_risk_clients']]
                        labels = ['High Risk', 'Medium Risk', 'Low Risk']
                        colors = ['red', 'orange', 'green']
                        
                        axes[i].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                        axes[i].set_title(f'{col} Risk Distribution')
                
                plt.tight_layout()
                return fig, axes
            return None, None

        result = self.generate_plot_safely(plot_risk_analysis)
        if result is not None and result[0] is not None:
            fig, _ = result
            img_path = os.path.join(self.output_folder, f"{table_name}_risk_analysis.png")
            plt.savefig(img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            image_paths.append(("Risk Analysis", img_path))

        results['image_paths'] = image_paths
        self.interpret_results("Leverage and Risk Analysis", results, table_name)
        self.technique_counter += 1

    def liquidity_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Liquidity Analysis"))
        
        # Identify liquidity-related columns
        liquidity_cols = [col for col in df.columns if any(keyword in col.upper() for keyword in [
            'CASH', 'WORKING_CAPITAL', 'CURRENT', 'ACCOUNTS_RECEIVABLE', 'ACCOUNTS_PAYABLE'
        ])]
        
        results = {"liquidity_metrics": {}}
        image_paths = []
        
        for col in liquidity_cols:
            if col in df.columns:
                liquidity_data = pd.to_numeric(df[col], errors='coerce').dropna()
                
                results["liquidity_metrics"][col] = {
                    'mean': float(liquidity_data.mean()),
                    'median': float(liquidity_data.median()),
                    'std_dev': float(liquidity_data.std()),
                    'positive_clients': int((liquidity_data > 0).sum()),
                    'negative_clients': int((liquidity_data <= 0).sum()),
                    'positive_percentage': float(((liquidity_data > 0).sum() / len(liquidity_data)) * 100)
                }

        # Create liquidity visualization
        def plot_liquidity_analysis():
            if len(liquidity_cols) >= 1:
                available_cols = [col for col in liquidity_cols if col in df.columns][:3]
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
                
                # Histogram of main liquidity metric
                main_col = available_cols[0]
                liquidity_data = pd.to_numeric(df[main_col], errors='coerce').dropna()
                
                ax1.hist(liquidity_data, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
                ax1.axvline(0, color='red', linestyle='--', label='Zero Line')
                ax1.set_title(f'Distribution of {main_col}')
                ax1.set_xlabel(main_col)
                ax1.set_ylabel('Number of Clients')
                ax1.legend()
                
                # Comparison of liquidity metrics if multiple available
                if len(available_cols) > 1:
                    for i, col in enumerate(available_cols):
                        col_data = pd.to_numeric(df[col], errors='coerce').dropna()
                        ax2.hist(col_data, alpha=0.5, label=col, bins=15)
                    
                    ax2.set_title('Liquidity Metrics Comparison')
                    ax2.set_xlabel('Values')
                    ax2.set_ylabel('Frequency')
                    ax2.legend()
                else:
                    # Show positive vs negative breakdown
                    positive_count = (liquidity_data > 0).sum()
                    negative_count = (liquidity_data <= 0).sum()
                    
                    ax2.pie([positive_count, negative_count], 
                           labels=['Positive', 'Negative/Zero'], 
                           colors=['green', 'red'],
                           autopct='%1.1f%%')
                    ax2.set_title(f'{main_col} - Positive vs Negative')
                
                plt.tight_layout()
                return fig, (ax1, ax2)
            return None, None

        result = self.generate_plot_safely(plot_liquidity_analysis)
        if result is not None and result[0] is not None:
            fig, _ = result
            img_path = os.path.join(self.output_folder, f"{table_name}_liquidity_analysis.png")
            plt.savefig(img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            image_paths.append(("Liquidity Analysis", img_path))

        results['image_paths'] = image_paths
        self.interpret_results("Liquidity Analysis", results, table_name)
        self.technique_counter += 1

    def efficiency_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Efficiency Analysis"))
        
        # Identify efficiency-related columns
        efficiency_cols = [col for col in df.columns if any(keyword in col.upper() for keyword in [
            'TURNOVER', 'DAYS', 'INVENTORY', 'RECEIVABLE', 'PAYABLE', 'OPERATING_OUTPUT'
        ])]
        
        results = {"efficiency_metrics": {}}
        image_paths = []
        
        for col in efficiency_cols:
            if col in df.columns:
                efficiency_data = pd.to_numeric(df[col], errors='coerce').dropna()
                
                results["efficiency_metrics"][col] = {
                    'mean': float(efficiency_data.mean()),
                    'median': float(efficiency_data.median()),
                    'std_dev': float(efficiency_data.std()),
                    'top_quartile': float(efficiency_data.quantile(0.75)),
                    'bottom_quartile': float(efficiency_data.quantile(0.25)),
                    'efficiency_score': float((efficiency_data > efficiency_data.median()).sum() / len(efficiency_data) * 100)
                }

        def plot_efficiency_analysis():
            if len(efficiency_cols) >= 1:
                available_cols = [col for col in efficiency_cols if col in df.columns][:4]
                
                fig, axes = plt.subplots(2, 2, figsize=(20, 16))
                axes = axes.flatten()
                
                for i, col in enumerate(available_cols[:4]):
                    efficiency_data = pd.to_numeric(df[col], errors='coerce').dropna()
                    
                    # Box plot for each efficiency metric
                    axes[i].boxplot(efficiency_data)
                    axes[i].set_title(f'{col} Distribution')
                    axes[i].set_ylabel(col)
                    
                    # Add quartile information
                    q1 = efficiency_data.quantile(0.25)
                    q3 = efficiency_data.quantile(0.75)
                    median = efficiency_data.median()
                    axes[i].text(0.7, 0.95, f'Q1: {q1:.2f}\nMedian: {median:.2f}\nQ3: {q3:.2f}', 
                                transform=axes[i].transAxes, verticalalignment='top',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
                
                plt.tight_layout()
                return fig, axes
            return None, None

        result = self.generate_plot_safely(plot_efficiency_analysis)
        if result is not None and result[0] is not None:
            fig, _ = result
            img_path = os.path.join(self.output_folder, f"{table_name}_efficiency_analysis.png")
            plt.savefig(img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            image_paths.append(("Efficiency Analysis", img_path))

        results['image_paths'] = image_paths
        self.interpret_results("Efficiency Analysis", results, table_name)
        self.technique_counter += 1

    def client_segmentation_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Client Segmentation Analysis"))
        
        # Select key financial metrics for segmentation
        key_metrics = [col for col in df.columns if any(keyword in col.upper() for keyword in [
            'TOTAL_ASSETS', 'TURNOVER', 'EBITDA', 'ROA', 'LEVERAGE'
        ])]
        
        results = {"segmentation_analysis": {}}
        image_paths = []
        
        if len(key_metrics) >= 2:
            # Prepare data for clustering
            segmentation_data = df[key_metrics].copy()
            segmentation_data = segmentation_data.apply(pd.to_numeric, errors='coerce').dropna()
            
            if len(segmentation_data) > 3:
                # Standardize the data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(segmentation_data)
                
                # Perform K-means clustering
                optimal_clusters = min(5, len(segmentation_data) // 3)
                kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(scaled_data)
                
                # Analyze each cluster
                segmentation_data['Cluster'] = cluster_labels
                cluster_profiles = {}
                
                for cluster_id in range(optimal_clusters):
                    cluster_data = segmentation_data[segmentation_data['Cluster'] == cluster_id]
                    
                    cluster_profile = {
                        'client_count': len(cluster_data),
                        'percentage': float((len(cluster_data) / len(segmentation_data)) * 100),
                        'characteristics': {}
                    }
                    
                    for metric in key_metrics:
                        if metric in cluster_data.columns:
                            cluster_mean = cluster_data[metric].mean()
                            overall_mean = segmentation_data[metric].mean()
                            cluster_profile['characteristics'][metric] = {
                                'cluster_average': float(cluster_mean),
                                'vs_portfolio_avg': float(((cluster_mean - overall_mean) / overall_mean) * 100)
                            }
                    
                    cluster_profiles[f'Cluster_{cluster_id}'] = cluster_profile
                
                results["segmentation_analysis"] = {
                    'total_clusters': optimal_clusters,
                    'cluster_profiles': cluster_profiles,
                    'metrics_used': key_metrics
                }

                def plot_segmentation_analysis():
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
                    
                    # Cluster size pie chart
                    cluster_sizes = [profile['client_count'] for profile in cluster_profiles.values()]
                    cluster_labels_list = list(cluster_profiles.keys())
                    
                    ax1.pie(cluster_sizes, labels=cluster_labels_list, autopct='%1.1f%%', startangle=90)
                    ax1.set_title('Client Distribution by Segment')
                    
                    # PCA visualization if we have enough dimensions
                    if len(key_metrics) >= 2:
                        pca = PCA(n_components=2)
                        pca_data = pca.fit_transform(scaled_data)
                        
                        scatter = ax2.scatter(pca_data[:, 0], pca_data[:, 1], c=cluster_labels, cmap='viridis')
                        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
                        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
                        ax2.set_title('Client Segments (PCA Visualization)')
                        plt.colorbar(scatter, ax=ax2)
                    
                    plt.tight_layout()
                    return fig, (ax1, ax2)

                result = self.generate_plot_safely(plot_segmentation_analysis)
                if result is not None and result[0] is not None:
                    fig, _ = result
                    img_path = os.path.join(self.output_folder, f"{table_name}_segmentation_analysis.png")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    image_paths.append(("Segmentation Analysis", img_path))

        results['image_paths'] = image_paths
        self.interpret_results("Client Segmentation Analysis", results, table_name)
        self.technique_counter += 1

    def risk_assessment_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Risk Assessment Analysis"))
        
        # Define risk indicators and their thresholds
        risk_indicators = {
            'LEVERAGE': {'high_risk': 3.0, 'medium_risk': 2.0, 'direction': 'higher_worse'},
            'DSCR': {'high_risk': 1.0, 'medium_risk': 1.25, 'direction': 'lower_worse'},
            'ROA': {'high_risk': 0.05, 'medium_risk': 0.10, 'direction': 'lower_worse'},
            'WORKING_CAPITAL_CURRENT_YEAR': {'high_risk': 0, 'medium_risk': 100000, 'direction': 'lower_worse'}
        }
        
        results = {"risk_assessment": {}}
        image_paths = []
        
        # Calculate overall risk scores for each client
        risk_scores = []
        risk_details = []
        
        for index, row in df.iterrows():
            client_risk_score = 0
            client_risk_factors = []
            
            for indicator, thresholds in risk_indicators.items():
                if indicator in df.columns:
                    value = pd.to_numeric(row[indicator], errors='coerce')
                    if pd.notna(value):
                        if thresholds['direction'] == 'higher_worse':
                            if value > thresholds['high_risk']:
                                client_risk_score += 3
                                client_risk_factors.append(f"High {indicator}: {value:.2f}")
                            elif value > thresholds['medium_risk']:
                                client_risk_score += 2
                                client_risk_factors.append(f"Medium {indicator}: {value:.2f}")
                            else:
                                client_risk_score += 1
                        else:  # lower_worse
                            if value < thresholds['high_risk']:
                                client_risk_score += 3
                                client_risk_factors.append(f"Low {indicator}: {value:.2f}")
                            elif value < thresholds['medium_risk']:
                                client_risk_score += 2
                                client_risk_factors.append(f"Medium {indicator}: {value:.2f}")
                            else:
                                client_risk_score += 1
            
            risk_scores.append(client_risk_score)
            risk_details.append(client_risk_factors)
        
        # Categorize clients by risk level
        risk_scores = np.array(risk_scores)
        high_risk_threshold = np.percentile(risk_scores, 80)
        medium_risk_threshold = np.percentile(risk_scores, 60)
        
        high_risk_clients = (risk_scores >= high_risk_threshold).sum()
        medium_risk_clients = ((risk_scores >= medium_risk_threshold) & (risk_scores < high_risk_threshold)).sum()
        low_risk_clients = (risk_scores < medium_risk_threshold).sum()
        
        results["risk_assessment"] = {
            'total_clients_assessed': len(risk_scores),
            'high_risk_clients': int(high_risk_clients),
            'medium_risk_clients': int(medium_risk_clients),
            'low_risk_clients': int(low_risk_clients),
            'high_risk_percentage': float((high_risk_clients / len(risk_scores)) * 100),
            'average_risk_score': float(np.mean(risk_scores)),
            'risk_score_std': float(np.std(risk_scores)),
            'risk_indicators_used': list(risk_indicators.keys())
        }

        def plot_risk_assessment():
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Risk distribution pie chart
            sizes = [high_risk_clients, medium_risk_clients, low_risk_clients]
            labels = ['High Risk', 'Medium Risk', 'Low Risk']
            colors = ['red', 'orange', 'green']
            
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Client Risk Distribution')
            
            # Risk score histogram
            ax2.hist(risk_scores, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            ax2.axvline(high_risk_threshold, color='red', linestyle='--', label=f'High Risk Threshold: {high_risk_threshold:.1f}')
            ax2.axvline(medium_risk_threshold, color='orange', linestyle='--', label=f'Medium Risk Threshold: {medium_risk_threshold:.1f}')
            ax2.set_title('Distribution of Risk Scores')
            ax2.set_xlabel('Risk Score')
            ax2.set_ylabel('Number of Clients')
            ax2.legend()
            
            plt.tight_layout()
            return fig, (ax1, ax2)

        result = self.generate_plot_safely(plot_risk_assessment)
        if result is not None and result[0] is not None:
            fig, _ = result
            img_path = os.path.join(self.output_folder, f"{table_name}_risk_assessment.png")
            plt.savefig(img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            image_paths.append(("Risk Assessment", img_path))

        results['image_paths'] = image_paths
        self.interpret_results("Risk Assessment Analysis", results, table_name)
        self.technique_counter += 1

    def correlation_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Correlation Analysis"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        # Convert any non-numeric columns to numeric
        for col in df.columns:
            if col not in numerical_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        results = {}
        image_paths = []
        
        if len(numerical_columns) > 1:
            correlation_matrix = df[numerical_columns].corr()
            
            # Find strongest correlations
            correlations = []
            for i in range(len(numerical_columns)):
                for j in range(i+1, len(numerical_columns)):
                    col1 = numerical_columns[i]
                    col2 = numerical_columns[j]
                    corr_value = correlation_matrix.loc[col1, col2]
                    if abs(corr_value) > 0.3:  # Only significant correlations
                        correlations.append({
                            'metric_1': col1,
                            'metric_2': col2,
                            'correlation': float(corr_value),
                            'strength': 'Strong' if abs(corr_value) > 0.7 else 'Moderate'
                        })
            
            # Sort by absolute correlation value
            correlations = sorted(correlations, key=lambda x: abs(x['correlation']), reverse=True)
            
            results = {
                'significant_correlations': correlations,
                'total_metrics': len(numerical_columns),
                'correlation_matrix_size': f"{len(numerical_columns)}x{len(numerical_columns)}"
            }

            def plot_correlation_heatmap():
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))
                
                # Full correlation matrix
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                           center=0, ax=ax1, fmt='.2f', square=True)
                ax1.set_title('Financial Metrics Correlation Matrix')
                
                # Strong correlations only (>0.5)
                strong_corr_mask = (correlation_matrix.abs() > 0.5) & (correlation_matrix != 1.0)
                strong_corr_features = strong_corr_mask.any().index[strong_corr_mask.any()]
                
                if len(strong_corr_features) > 1:
                    strong_corr_matrix = correlation_matrix.loc[strong_corr_features, strong_corr_features]
                    sns.heatmap(strong_corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                               center=0, ax=ax2, fmt='.2f', square=True)
                    ax2.set_title('Strong Correlations (|r| > 0.5)')
                else:
                    ax2.text(0.5, 0.5, 'No strong correlations found\n(|r| > 0.5)', 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax2.transAxes, fontsize=14)
                    ax2.set_title('Strong Correlations (|r| > 0.5)')
                
                plt.tight_layout()
                return fig, (ax1, ax2)

            result = self.generate_plot_safely(plot_correlation_heatmap)
            if result is not None and result[0] is not None:
                fig, _ = result
                img_path = os.path.join(self.output_folder, f"{table_name}_correlation_analysis.png")
                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_paths.append(("Correlation Analysis", img_path))
        else:
            results = "N/A - Not enough numerical features for correlation analysis"
        
        results['image_paths'] = image_paths
        self.interpret_results("Correlation Analysis", results, table_name)
        self.technique_counter += 1

    def outlier_detection_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Outlier Detection Analysis"))
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        # Convert non-numeric columns
        for col in df.columns:
            if col not in numerical_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        results = {"outlier_analysis": {}}
        image_paths = []
        
        # Key financial metrics to focus on
        key_metrics = [col for col in numerical_columns if any(keyword in col.upper() for keyword in [
            'TOTAL_ASSETS', 'TURNOVER', 'EBITDA', 'LEVERAGE', 'ROA'
        ])][:4]  # Limit to 4 key metrics
        
        for col in key_metrics:
            if col in df.columns:
                col_data = pd.to_numeric(df[col], errors='coerce').dropna()
                
                # Calculate outliers using IQR method
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                
                # Calculate Z-scores for extreme outliers
                z_scores = np.abs(zscore(col_data))
                extreme_outliers = col_data[z_scores > 3]
                
                results["outlier_analysis"][col] = {
                    'total_outliers': len(outliers),
                    'outlier_percentage': float((len(outliers) / len(col_data)) * 100),
                    'extreme_outliers': len(extreme_outliers),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'outlier_values': {
                        'highest': outliers.nlargest(3).tolist() if len(outliers) > 0 else [],
                        'lowest': outliers.nsmallest(3).tolist() if len(outliers) > 0 else []
                    }
                }

        def plot_outlier_detection():
            if len(key_metrics) >= 1:
                fig, axes = plt.subplots(2, 2, figsize=(20, 16))
                axes = axes.flatten()
                
                for i, col in enumerate(key_metrics[:4]):
                    if i < 4:
                        col_data = pd.to_numeric(df[col], errors='coerce').dropna()
                        
                        # Box plot
                        axes[i].boxplot(col_data)
                        axes[i].set_title(f'{col} - Outlier Detection')
                        axes[i].set_ylabel(col)
                        
                        # Add statistics
                        if col in results["outlier_analysis"]:
                            outlier_info = results["outlier_analysis"][col]
                            axes[i].text(0.02, 0.98, 
                                        f'Outliers: {outlier_info["total_outliers"]} ({outlier_info["outlier_percentage"]:.1f}%)',
                                        transform=axes[i].transAxes, verticalalignment='top',
                                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
                
                plt.tight_layout()
                return fig, axes
            return None, None

        result = self.generate_plot_safely(plot_outlier_detection)
        if result is not None and result[0] is not None:
            fig, _ = result
            img_path = os.path.join(self.output_folder, f"{table_name}_outlier_detection.png")
            plt.savefig(img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            image_paths.append(("Outlier Detection", img_path))

        results['image_paths'] = image_paths
        self.interpret_results("Outlier Detection Analysis", results, table_name)
        self.technique_counter += 1

    def benchmarking_analysis(self, df, table_name):
        print(info(f"Performing test {self.technique_counter}/{self.total_techniques} - Benchmarking Analysis"))
        
        # Define industry benchmarks (these can be customized based on industry)
        industry_benchmarks = {
            'ROA': {'excellent': 0.15, 'good': 0.10, 'average': 0.05, 'poor': 0.02},
            'LEVERAGE': {'excellent': 1.5, 'good': 2.0, 'average': 3.0, 'poor': 4.0},
            'DSCR': {'excellent': 2.0, 'good': 1.5, 'average': 1.25, 'poor': 1.0}
        }
        
        results = {"benchmarking_analysis": {}}
        image_paths = []
        
        for metric, benchmarks in industry_benchmarks.items():
            if metric in df.columns:
                metric_data = pd.to_numeric(df[metric], errors='coerce').dropna()
                
                # Categorize clients based on benchmarks
                if metric in ['ROA', 'DSCR']:  # Higher is better
                    excellent = (metric_data >= benchmarks['excellent']).sum()
                    good = ((metric_data >= benchmarks['good']) & (metric_data < benchmarks['excellent'])).sum()
                    average = ((metric_data >= benchmarks['average']) & (metric_data < benchmarks['good'])).sum()
                    poor = (metric_data < benchmarks['average']).sum()
                else:  # Lower is better (LEVERAGE)
                    excellent = (metric_data <= benchmarks['excellent']).sum()
                    good = ((metric_data > benchmarks['excellent']) & (metric_data <= benchmarks['good'])).sum()
                    average = ((metric_data > benchmarks['good']) & (metric_data <= benchmarks['average'])).sum()
                    poor = (metric_data > benchmarks['average']).sum()
                
                results["benchmarking_analysis"][metric] = {
                    'excellent_performers': int(excellent),
                    'good_performers': int(good),
                    'average_performers': int(average),
                    'poor_performers': int(poor),
                    'excellent_percentage': float((excellent / len(metric_data)) * 100),
                    'below_average_percentage': float((poor / len(metric_data)) * 100),
                    'portfolio_average': float(metric_data.mean()),
                    'benchmarks_used': benchmarks
                }

        def plot_benchmarking_analysis():
            if len(results["benchmarking_analysis"]) > 0:
                metrics = list(results["benchmarking_analysis"].keys())
                fig, axes = plt.subplots(1, len(metrics), figsize=(8*len(metrics), 8))
                
                if len(metrics) == 1:
                    axes = [axes]
                
                for i, metric in enumerate(metrics):
                    benchmark_data = results["benchmarking_analysis"][metric]
                    
                    # Performance distribution pie chart
                    sizes = [
                        benchmark_data['excellent_performers'],
                        benchmark_data['good_performers'],
                        benchmark_data['average_performers'],
                        benchmark_data['poor_performers']
                    ]
                    labels = ['Excellent', 'Good', 'Average', 'Poor']
                    colors = ['darkgreen', 'lightgreen', 'orange', 'red']
                    
                    axes[i].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                    axes[i].set_title(f'{metric} Performance vs Industry Benchmarks')
                
                plt.tight_layout()
                return fig, axes
            return None, None

        result = self.generate_plot_safely(plot_benchmarking_analysis)
        if result is not None and result[0] is not None:
            fig, _ = result
            img_path = os.path.join(self.output_folder, f"{table_name}_benchmarking_analysis.png")
            plt.savefig(img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            image_paths.append(("Benchmarking Analysis", img_path))

        results['image_paths'] = image_paths
        self.interpret_results("Benchmarking Analysis", results, table_name)
        self.technique_counter += 1

    def interpret_results(self, analysis_type, results, table_name):
        # Format results for interpretation
        if isinstance(results, dict):
            results_str = ""
            for key, value in results.items():
                if key != 'image_paths':
                    if isinstance(value, dict):
                        results_str += f"\n{key}:\n"
                        for sub_key, sub_value in value.items():
                            results_str += f"  {sub_key}: {sub_value}\n"
                    else:
                        results_str += f"{key}: {value}\n"
        else:
            results_str = str(results)

        num_visualizations = len(results.get('image_paths', []))
        results_str += f"\nNumber of visualizations created: {num_visualizations}"

        worker_prompt = f"""
        You are an expert financial analyst providing insights on client portfolio analysis results. Your task is to interpret the following financial analysis results and provide detailed, data-driven insights for portfolio management and risk assessment.

        Analysis type: {analysis_type}
        Portfolio table: {table_name}
        Business context: {self.database_description}

        Financial Analysis Results:
        {results_str}

        Please provide a comprehensive interpretation focusing on:
        1. Client portfolio insights and financial health patterns
        2. Risk assessment and risk management implications
        3. Business opportunities and areas requiring attention
        4. Specific clients or segments that warrant immediate action

        Structure your response as follows:

        1. Key Financial Insights:
        [Provide 3-4 critical insights about the client portfolio's financial health, including specific percentages and figures]

        2. Risk Assessment:
        [Identify high-risk clients or concerning patterns, with quantified risk levels and specific metrics]

        3. Performance Analysis:
        [Analyze client performance distribution, outliers, and benchmark comparisons with concrete numbers]

        4. Portfolio Management Recommendations:
        [Provide actionable recommendations for portfolio management, client relationship strategies, and risk mitigation]

        Use actual metric names and values from the data. Focus on actionable insights that can improve portfolio performance and risk management.

        Financial Analysis:
        """

        worker_interpretation = self.worker_erag_api.chat([
            {"role": "system", "content": "You are an expert financial analyst specializing in portfolio management and risk assessment. Provide actionable insights based on client financial data."},
            {"role": "user", "content": worker_prompt}
        ])

        supervisor_prompt = f"""
        As a senior portfolio manager and risk officer, review and enhance this financial analysis interpretation:

        Analysis type: {analysis_type}
        Portfolio context: {self.database_description}

        Financial Data:
        {results_str}

        Previous Analysis:
        {worker_interpretation}

        Provide an enhanced business-focused interpretation that includes:

        1. Executive Summary:
        [Concise overview of portfolio health and key concerns]

        2. Strategic Implications:
        [How these findings impact overall portfolio strategy and client relationship management]

        3. Risk Management Priorities:
        [Immediate actions needed for risk mitigation, ranked by priority]

        4. Business Development Opportunities:
        [Opportunities to improve client relationships, cross-selling, or portfolio optimization]

        Focus on actionable business decisions and strategic portfolio management. Include specific recommendations with measurable targets where possible.

        Enhanced Business Analysis:
        """

        supervisor_analysis = self.supervisor_erag_api.chat([
            {"role": "system", "content": "You are a senior portfolio manager providing strategic business analysis based on financial data. Focus on actionable business decisions and portfolio optimization."},
            {"role": "user", "content": supervisor_prompt}
        ])

        combined_interpretation = f"""
        Financial Analysis:
        {worker_interpretation.strip()}

        Strategic Business Analysis:
        {supervisor_analysis.strip()}
        """

        print(success(f"Combined Financial Interpretation for {analysis_type}:"))
        print(combined_interpretation.strip())

        self.text_output += f"\n{combined_interpretation.strip()}\n\n"

        # Handle images for the PDF report
        image_data = []
        if isinstance(results, dict) and 'image_paths' in results:
            for img in results['image_paths']:
                if isinstance(img, tuple) and len(img) == 2:
                    image_data.append(img)
                elif isinstance(img, str):
                    image_data.append((analysis_type, img))

        self.pdf_content.append((analysis_type, image_data, combined_interpretation.strip()))
        self.image_data.extend(image_data)

        # Extract important findings
        lines = combined_interpretation.strip().split('\n')
        for i, line in enumerate(lines):
            if any(keyword in line for keyword in ["Key Financial Insights:", "Risk Assessment:", "Executive Summary:"]):
                for finding in lines[i+1:]:
                    if finding.strip() and not any(keyword in finding for keyword in ["Strategic Implications:", "Risk Management", "Business Development"]):
                        self.findings.append(f"{analysis_type}: {finding.strip()}")
                    elif any(keyword in finding for keyword in ["Strategic Implications:", "Risk Management", "Business Development"]):
                        break

    def generate_executive_summary(self):
        if not self.findings:
            self.executive_summary = "No significant financial findings were identified during the analysis. This could indicate a well-balanced portfolio or require further investigation into data quality and completeness."
            return

        summary_prompt = f"""
        As an expert financial analyst and portfolio manager, provide an executive summary of the following client portfolio financial analysis findings:
        
        Portfolio Context: {self.database_description}
        
        Key Findings:
        {self.findings}
        
        The executive summary should:
        1. Provide a clear overview of the overall portfolio financial health
        2. Highlight the most critical financial risks and opportunities with specific figures
        3. Identify immediate actions required for portfolio management and risk mitigation
        4. Discuss client segments requiring special attention or management strategies
        5. Conclude with strategic recommendations for portfolio optimization and growth

        Structure the summary in multiple paragraphs for readability.
        Focus on actionable insights that support executive decision-making and portfolio strategy.
        Include specific percentages, client counts, and financial metrics where available.
        """
        
        try:
            worker_summary = self.worker_erag_api.chat([
                {"role": "system", "content": "You are an expert financial analyst providing an executive summary for portfolio management. Focus on strategic insights and actionable recommendations."},
                {"role": "user", "content": summary_prompt}
            ])
            
            if worker_summary is not None:
                supervisor_prompt = f"""
                As a senior executive and portfolio strategist, review and enhance this financial portfolio executive summary:

                {worker_summary}

                Enhance the summary by:
                1. Ensuring it addresses key executive concerns about portfolio performance and risk
                2. Making recommendations more specific and actionable with clear timelines
                3. Adding strategic context about portfolio positioning and competitive advantages
                4. Emphasizing the most critical business decisions that need to be made
                5. Including specific metrics and targets for portfolio optimization

                Provide a refined executive summary that would be appropriate for board-level discussions and strategic planning.
                """

                enhanced_summary = self.supervisor_erag_api.chat([
                    {"role": "system", "content": "You are a senior executive providing strategic portfolio analysis for executive decision-making. Focus on high-level business strategy and actionable recommendations."},
                    {"role": "user", "content": supervisor_prompt}
                ])

                self.executive_summary = enhanced_summary.strip()
            else:
                self.executive_summary = "Error: Unable to generate financial executive summary."
        except Exception as e:
            print(error(f"An error occurred while generating the financial executive summary: {str(e)}"))
            self.executive_summary = "Error: Unable to generate financial executive summary due to an exception."

        print(success("Enhanced Financial Executive Summary generated successfully."))
        print(self.executive_summary)

    def save_text_output(self):
        output_file = os.path.join(self.output_folder, "fxda_results.txt")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(self.text_output)

    def generate_pdf_report(self):
        try:
            report_title = "Financial Portfolio Analysis Report"
            pdf_generator = PDFReportGenerator(self.output_folder, self.llm_name, "Financial Portfolio Analysis")
            
            pdf_file = pdf_generator.create_enhanced_pdf_report(
                self.executive_summary,
                self.findings,
                self.pdf_content,
                self.image_data,
                filename="fxda_report",
                report_title=report_title
            )
            
            if pdf_file:
                print(success(f"Financial PDF report generated successfully: {pdf_file}"))
                return pdf_file
            else:
                print(error("Failed to generate financial PDF report"))
                return None
        
        except Exception as e:
            error_message = f"An error occurred while generating the PDF report: {str(e)}"
            print(error(error_message))
            return None

    def calculate_figure_size(self, aspect_ratio=16/9):
        max_width = int(np.sqrt(self.max_pixels * aspect_ratio))
        max_height = int(max_width / aspect_ratio)
        return (max_width / 100, max_height / 100)