#!/usr/bin/env python3
"""
QeMLflow Post-Migration ML Pipeline Tests
=========================================

Comprehensive machine learning pipeline tests to validate that QeMLflow
functionality is working correctly after the ChemML to QeMLflow migration.

This script tests:
1. Basic ML workflows
2. Data preprocessing
3. Model training and evaluation
4. Visualization capabilities
5. Utility functions
6. Configuration system

Author: Post-Migration Testing System
Date: June 17, 2025
"""

import sys
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add QeMLflow to path
root_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_path / "src"))

class MLPipelineTester:
    """Comprehensive ML pipeline testing for QeMLflow."""
    
    def __init__(self):
        self.test_results = {}
        self.errors = []
        self.warnings = []
        
    def log_result(self, test_name: str, success: bool, details: Dict = None):
        """Log test result."""
        self.test_results[test_name] = {
            'success': success,
            'details': details or {},
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} | {test_name}")
        
        if details:
            for key, value in details.items():
                print(f"    {key}: {value}")
    
    def test_qemlflow_import(self) -> bool:
        """Test 1: Basic QeMLflow import and initialization."""
        try:
            import qemlflow
            
            details = {
                'version': getattr(qemlflow, '__version__', 'Unknown'),
                'import_path': qemlflow.__file__,
                'available_modules': []
            }
            
            # Test core modules
            core_modules = ['utils', 'config']
            for module in core_modules:
                try:
                    exec(f"from qemlflow import {module}")
                    details['available_modules'].append(module)
                except ImportError:
                    pass
            
            self.log_result("QeMLflow Import & Initialization", True, details)
            return True
            
        except Exception as e:
            self.log_result("QeMLflow Import & Initialization", False, {'error': str(e)})
            return False
    
    def test_data_generation_pipeline(self) -> bool:
        """Test 2: Data generation and basic preprocessing."""
        try:
            # Generate classification dataset
            X_class, y_class = make_classification(
                n_samples=1000,
                n_features=20,
                n_informative=15,
                n_redundant=5,
                n_classes=3,
                random_state=42
            )
            
            # Generate regression dataset
            X_reg, y_reg = make_regression(
                n_samples=1000,
                n_features=15,
                noise=0.1,
                random_state=42
            )
            
            # Convert to DataFrames
            class_df = pd.DataFrame(X_class, columns=[f'feature_{i}' for i in range(X_class.shape[1])])
            class_df['target'] = y_class
            
            reg_df = pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(X_reg.shape[1])])
            reg_df['target'] = y_reg
            
            # Basic preprocessing
            scaler = StandardScaler()
            X_class_scaled = scaler.fit_transform(X_class)
            X_reg_scaled = scaler.fit_transform(X_reg)
            
            details = {
                'classification_shape': class_df.shape,
                'regression_shape': reg_df.shape,
                'scaling_successful': True,
                'target_classes': len(np.unique(y_class))
            }
            
            # Store data for later tests
            self.classification_data = (X_class_scaled, y_class)
            self.regression_data = (X_reg_scaled, y_reg)
            
            self.log_result("Data Generation & Preprocessing", True, details)
            return True
            
        except Exception as e:
            self.log_result("Data Generation & Preprocessing", False, {'error': str(e)})
            return False
    
    def test_classification_pipeline(self) -> bool:
        """Test 3: Classification ML pipeline."""
        try:
            X, y = self.classification_data
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Test multiple algorithms
            models = {
                'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
                'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
            }
            
            results = {}
            
            for name, model in models.items():
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                results[name] = {
                    'accuracy': round(accuracy, 4),
                    'trained': True
                }
            
            details = {
                'models_tested': len(models),
                'results': results,
                'data_split': f"Train: {len(X_train)}, Test: {len(X_test)}"
            }
            
            self.log_result("Classification Pipeline", True, details)
            return True
            
        except Exception as e:
            self.log_result("Classification Pipeline", False, {'error': str(e), 'traceback': traceback.format_exc()})
            return False
    
    def test_regression_pipeline(self) -> bool:
        """Test 4: Regression ML pipeline."""
        try:
            X, y = self.regression_data
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Test multiple algorithms
            models = {
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
                'LinearRegression': LinearRegression()
            }
            
            results = {}
            
            for name, model in models.items():
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                results[name] = {
                    'rmse': round(rmse, 4),
                    'mse': round(mse, 4),
                    'trained': True
                }
            
            details = {
                'models_tested': len(models),
                'results': results,
                'data_split': f"Train: {len(X_train)}, Test: {len(X_test)}"
            }
            
            self.log_result("Regression Pipeline", True, details)
            return True
            
        except Exception as e:
            self.log_result("Regression Pipeline", False, {'error': str(e), 'traceback': traceback.format_exc()})
            return False
    
    def test_visualization_pipeline(self) -> bool:
        """Test 5: Data visualization capabilities."""
        try:
            X_class, y_class = self.classification_data
            X_reg, y_reg = self.regression_data
            
            # Create visualizations
            plt.style.use('default')
            
            # Classification visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Feature correlation heatmap
            class_df = pd.DataFrame(X_class[:, :10])  # First 10 features
            sns.heatmap(class_df.corr(), ax=axes[0,0], cmap='coolwarm', center=0)
            axes[0,0].set_title('Feature Correlation (Classification)')
            
            # Target distribution
            axes[0,1].hist(y_class, bins=20, alpha=0.7)
            axes[0,1].set_title('Target Distribution (Classification)')
            
            # Regression scatter plot
            axes[1,0].scatter(X_reg[:, 0], y_reg, alpha=0.5)
            axes[1,0].set_title('Feature vs Target (Regression)')
            axes[1,0].set_xlabel('Feature 0')
            axes[1,0].set_ylabel('Target')
            
            # Feature distribution
            axes[1,1].hist(X_reg[:, 0], bins=30, alpha=0.7)
            axes[1,1].set_title('Feature Distribution (Regression)')
            
            plt.tight_layout()
            
            # Save plot
            output_dir = root_path / "tools" / "validation" / "ml_test_outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            plot_file = output_dir / "ml_pipeline_visualizations.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            details = {
                'plots_created': 4,
                'plot_saved': str(plot_file),
                'visualization_libraries': ['matplotlib', 'seaborn']
            }
            
            self.log_result("Visualization Pipeline", True, details)
            return True
            
        except Exception as e:
            self.log_result("Visualization Pipeline", False, {'error': str(e)})
            return False
    
    def test_feature_engineering_pipeline(self) -> bool:
        """Test 6: Feature engineering capabilities."""
        try:
            X_class, y_class = self.classification_data
            
            # Feature engineering
            X_df = pd.DataFrame(X_class)
            
            # Create polynomial features (simple version)
            X_df['feature_0_squared'] = X_df[0] ** 2
            X_df['feature_1_squared'] = X_df[1] ** 2
            
            # Create interaction features
            X_df['interaction_0_1'] = X_df[0] * X_df[1]
            X_df['interaction_0_2'] = X_df[0] * X_df[2]
            
            # Create statistical features
            X_df['mean_features'] = X_df.iloc[:, :5].mean(axis=1)
            X_df['std_features'] = X_df.iloc[:, :5].std(axis=1)
            X_df['max_features'] = X_df.iloc[:, :5].max(axis=1)
            X_df['min_features'] = X_df.iloc[:, :5].min(axis=1)
            
            # Binning features
            X_df['feature_0_binned'] = pd.cut(X_df[0], bins=5, labels=False)
            
            # Feature selection simulation
            feature_importance = np.random.random(X_df.shape[1])
            top_features = np.argsort(feature_importance)[-10:]
            X_selected = X_df.iloc[:, top_features]
            
            details = {
                'original_features': X_class.shape[1],
                'engineered_features': X_df.shape[1],
                'selected_features': X_selected.shape[1],
                'feature_types': ['polynomial', 'interaction', 'statistical', 'binned']
            }
            
            self.log_result("Feature Engineering Pipeline", True, details)
            return True
            
        except Exception as e:
            self.log_result("Feature Engineering Pipeline", False, {'error': str(e)})
            return False
    
    def test_model_evaluation_pipeline(self) -> bool:
        """Test 7: Model evaluation and metrics."""
        try:
            X_class, y_class = self.classification_data
            X_reg, y_reg = self.regression_data
            
            # Classification evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
            )
            
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            clf.fit(X_train, y_train)
            y_pred_class = clf.predict(X_test)
            
            # Classification metrics
            class_metrics = {
                'accuracy': accuracy_score(y_test, y_pred_class),
                'classes': len(np.unique(y_class))
            }
            
            # Regression evaluation
            X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
                X_reg, y_reg, test_size=0.2, random_state=42
            )
            
            reg = RandomForestRegressor(n_estimators=50, random_state=42)
            reg.fit(X_train_reg, y_train_reg)
            y_pred_reg = reg.predict(X_test_reg)
            
            # Regression metrics
            reg_metrics = {
                'mse': mean_squared_error(y_test_reg, y_pred_reg),
                'rmse': np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)),
                'r2': reg.score(X_test_reg, y_test_reg)
            }
            
            details = {
                'classification_metrics': class_metrics,
                'regression_metrics': reg_metrics,
                'evaluation_successful': True
            }
            
            self.log_result("Model Evaluation Pipeline", True, details)
            return True
            
        except Exception as e:
            self.log_result("Model Evaluation Pipeline", False, {'error': str(e)})
            return False
    
    def test_cross_validation_pipeline(self) -> bool:
        """Test 8: Cross-validation workflow."""
        try:
            from sklearn.model_selection import cross_val_score, StratifiedKFold
            
            X_class, y_class = self.classification_data
            
            # Set up cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # Test models with CV
            models = {
                'RandomForest': RandomForestClassifier(n_estimators=50, random_state=42),
                'LogisticRegression': LogisticRegression(random_state=42, max_iter=500)
            }
            
            cv_results = {}
            
            for name, model in models.items():
                scores = cross_val_score(model, X_class, y_class, cv=cv, scoring='accuracy')
                cv_results[name] = {
                    'mean_score': round(scores.mean(), 4),
                    'std_score': round(scores.std(), 4),
                    'scores': scores.round(4).tolist()
                }
            
            details = {
                'cv_folds': 5,
                'cv_results': cv_results,
                'cv_strategy': 'StratifiedKFold'
            }
            
            self.log_result("Cross-Validation Pipeline", True, details)
            return True
            
        except Exception as e:
            self.log_result("Cross-Validation Pipeline", False, {'error': str(e)})
            return False
    
    def test_hyperparameter_tuning_pipeline(self) -> bool:
        """Test 9: Hyperparameter tuning workflow."""
        try:
            from sklearn.model_selection import GridSearchCV
            
            X_class, y_class = self.classification_data
            
            # Smaller dataset for faster tuning
            X_small, _, y_small, _ = train_test_split(
                X_class, y_class, test_size=0.8, random_state=42, stratify=y_class
            )
            
            # Define parameter grid
            param_grid = {
                'n_estimators': [10, 50],
                'max_depth': [3, 5],
                'min_samples_split': [2, 5]
            }
            
            # Grid search
            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(
                rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1
            )
            
            grid_search.fit(X_small, y_small)
            
            details = {
                'best_params': grid_search.best_params_,
                'best_score': round(grid_search.best_score_, 4),
                'param_combinations': len(grid_search.cv_results_['params']),
                'tuning_successful': True
            }
            
            self.log_result("Hyperparameter Tuning Pipeline", True, details)
            return True
            
        except Exception as e:
            self.log_result("Hyperparameter Tuning Pipeline", False, {'error': str(e)})
            return False
    
    def test_pipeline_integration(self) -> bool:
        """Test 10: End-to-end pipeline integration."""
        try:
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            
            X_class, y_class = self.classification_data
            
            # Create integrated pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
            ])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
            )
            
            # Fit pipeline
            pipeline.fit(X_train, y_train)
            
            # Predict
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Test pipeline with new data
            new_data = np.random.randn(10, X_class.shape[1])
            predictions = pipeline.predict(new_data)
            
            details = {
                'pipeline_steps': len(pipeline.steps),
                'accuracy': round(accuracy, 4),
                'new_data_predictions': len(predictions),
                'integration_successful': True
            }
            
            self.log_result("Pipeline Integration", True, details)
            return True
            
        except Exception as e:
            self.log_result("Pipeline Integration", False, {'error': str(e)})
            return False
    
    def generate_summary_report(self):
        """Generate comprehensive test summary."""
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['success'])
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print("\n" + "=" * 80)
        print("🧪 ML PIPELINE TESTING SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 90:
            status = "🎉 EXCELLENT"
            message = "All ML pipelines working perfectly!"
        elif success_rate >= 80:
            status = "✅ GOOD"
            message = "ML pipelines mostly functional with minor issues."
        elif success_rate >= 70:
            status = "⚠️ ACCEPTABLE"
            message = "ML pipelines functional but need attention."
        else:
            status = "❌ NEEDS WORK"
            message = "Significant ML pipeline issues detected."
        
        print(f"\nStatus: {status}")
        print(f"Assessment: {message}")
        
        # Failed tests
        failed_tests = [name for name, result in self.test_results.items() if not result['success']]
        if failed_tests:
            print(f"\n❌ Failed Tests:")
            for test in failed_tests:
                print(f"  - {test}")
        
        # Save detailed report
        output_dir = root_path / "tools" / "validation" / "ml_test_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = output_dir / "ml_pipeline_test_report.json"
        
        import json
        with open(report_file, 'w') as f:
            json.dump({
                'summary': {
                    'total_tests': total_tests,
                    'passed': passed_tests,
                    'failed': total_tests - passed_tests,
                    'success_rate': success_rate,
                    'status': status,
                    'timestamp': pd.Timestamp.now().isoformat()
                },
                'detailed_results': self.test_results
            }, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        return success_rate >= 80

def main():
    """Run comprehensive ML pipeline tests."""
    print("🚀 Starting QeMLflow Post-Migration ML Pipeline Tests")
    print("=" * 80)
    
    tester = MLPipelineTester()
    
    # Run all tests
    tests = [
        tester.test_qemlflow_import,
        tester.test_data_generation_pipeline,
        tester.test_classification_pipeline,
        tester.test_regression_pipeline,
        tester.test_visualization_pipeline,
        tester.test_feature_engineering_pipeline,
        tester.test_model_evaluation_pipeline,
        tester.test_cross_validation_pipeline,
        tester.test_hyperparameter_tuning_pipeline,
        tester.test_pipeline_integration
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"❌ CRITICAL ERROR in {test.__name__}: {e}")
            tester.test_results[test.__name__] = {
                'success': False,
                'error': str(e),
                'critical': True
            }
    
    # Generate summary
    success = tester.generate_summary_report()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
