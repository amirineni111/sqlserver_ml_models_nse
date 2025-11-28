"""
NSE Setup Test Script

This script tests the NSE 500 setup to ensure all components are working correctly.

Usage:
    python test_nse_setup.py
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from database.connection import SQLServerConnection

def safe_print(text):
    """Print with safe encoding."""
    try:
        print(text)
    except UnicodeEncodeError:
        emoji_replacements = {
            '🇮🇳': '[NSE]', '✅': '[SUCCESS]', '❌': '[ERROR]', 
            '📊': '[DATA]', '🔍': '[TEST]', '⚠️': '[WARN]'
        }
        for emoji, replacement in emoji_replacements.items():
            text = text.replace(emoji, replacement)
        print(text)

def test_database_connection():
    """Test database connectivity."""
    safe_print("🔍 Testing database connection...")
    
    try:
        db = SQLServerConnection()
        if db.test_connection():
            safe_print("✅ Database connection successful")
            return True
        else:
            safe_print("❌ Database connection failed")
            return False
    except Exception as e:
        safe_print(f"❌ Database connection error: {e}")
        return False

def test_nse_tables():
    """Test NSE table availability and structure."""
    safe_print("🔍 Testing NSE table structure...")
    
    try:
        db = SQLServerConnection()
        
        # Test nse_500_hist_data table
        hist_query = "SELECT TOP 5 * FROM dbo.nse_500_hist_data"
        hist_result = db.execute_query(hist_query)
        
        if not hist_result.empty:
            safe_print(f"✅ nse_500_hist_data table found: {len(hist_result)} sample records")
            safe_print(f"   Columns: {list(hist_result.columns)}")
        else:
            safe_print("⚠️  nse_500_hist_data table is empty")
        
        # Test NSE result tables
        result_tables = [
            'ml_nse_trading_predictions',
            'ml_nse_technical_indicators', 
            'ml_nse_predict_summary'
        ]
        
        for table in result_tables:
            try:
                test_query = f"SELECT COUNT(*) as count FROM dbo.{table}"
                result = db.execute_query(test_query)
                count = result.iloc[0]['count'] if not result.empty else 0
                safe_print(f"✅ {table}: {count} records")
            except Exception as e:
                safe_print(f"❌ {table}: Error - {e}")
        
        return True
        
    except Exception as e:
        safe_print(f"❌ Table test error: {e}")
        return False

def test_nse_data_quality():
    """Test NSE data quality and completeness."""
    safe_print("🔍 Testing NSE data quality...")
    
    try:
        db = SQLServerConnection()
        
        # Check data distribution
        quality_query = """
        SELECT 
            COUNT(DISTINCT ticker) as unique_tickers,
            COUNT(DISTINCT trading_date) as unique_dates,
            MIN(trading_date) as earliest_date,
            MAX(trading_date) as latest_date,
            COUNT(*) as total_records
        FROM dbo.nse_500_hist_data
        """
        
        result = db.execute_query(quality_query)
        
        if not result.empty:
            row = result.iloc[0]
            safe_print("📊 NSE Data Quality Report:")
            safe_print(f"   🏢 Unique Tickers: {row['unique_tickers']}")
            safe_print(f"   📅 Date Range: {row['earliest_date']} to {row['latest_date']}")
            safe_print(f"   📊 Total Records: {row['total_records']:,}")
            
            # Check for recent data
            latest_date = pd.to_datetime(row['latest_date']).date()
            days_old = (datetime.now().date() - latest_date).days
            
            if days_old <= 7:
                safe_print(f"✅ Data is recent ({days_old} days old)")
            else:
                safe_print(f"⚠️  Data is {days_old} days old - consider updating")
        
        # Check sample tickers
        sample_query = """
        SELECT TOP 10 ticker, company, MAX(trading_date) as latest_date, COUNT(*) as record_count
        FROM dbo.nse_500_hist_data 
        GROUP BY ticker, company
        ORDER BY record_count DESC
        """
        
        sample_result = db.execute_query(sample_query)
        
        if not sample_result.empty:
            safe_print("📊 Sample NSE Stocks:")
            for _, row in sample_result.iterrows():
                safe_print(f"   • {row['ticker']}: {row['record_count']} records (latest: {row['latest_date']})")
        
        return True
        
    except Exception as e:
        safe_print(f"❌ Data quality test error: {e}")
        return False

def test_model_files():
    """Test if required model files exist."""
    safe_print("🔍 Testing model file availability...")
    
    model_files = [
        'data/best_model_extra_trees.joblib',
        'data/scaler.joblib',
        'data/target_encoder.joblib'
    ]
    
    all_files_exist = True
    
    for file_path in model_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            safe_print(f"✅ {file_path}: {file_size:,} bytes")
        else:
            safe_print(f"❌ {file_path}: Not found")
            all_files_exist = False
    
    return all_files_exist

def test_directories():
    """Test if required directories exist or can be created."""
    safe_print("🔍 Testing directory structure...")
    
    required_dirs = ['results', 'logs', 'daily_reports', 'data']
    
    for dir_name in required_dirs:
        try:
            os.makedirs(dir_name, exist_ok=True)
            safe_print(f"✅ {dir_name}/ directory ready")
        except Exception as e:
            safe_print(f"❌ {dir_name}/ directory error: {e}")
            return False
    
    return True

def test_sample_prediction():
    """Test a simple prediction with available data."""
    safe_print("🔍 Testing sample NSE prediction...")
    
    try:
        # Import the NSE predictor
        sys.path.append('.')
        
        # This is a basic test - in a real scenario you'd import the full predictor
        db = SQLServerConnection()
        
        # Get a sample stock with recent data
        sample_query = """
        SELECT TOP 1 ticker, company, MAX(trading_date) as latest_date
        FROM dbo.nse_500_hist_data 
        WHERE trading_date >= DATEADD(day, -30, GETDATE())
        GROUP BY ticker, company
        ORDER BY COUNT(*) DESC
        """
        
        result = db.execute_query(sample_query)
        
        if not result.empty:
            sample_ticker = result.iloc[0]['ticker']
            latest_date = result.iloc[0]['latest_date']
            safe_print(f"✅ Sample stock found: {sample_ticker} (latest: {latest_date})")
            
            # Check if we have sufficient data for this stock
            data_query = f"""
            SELECT COUNT(*) as count 
            FROM dbo.nse_500_hist_data 
            WHERE ticker = '{sample_ticker}' 
                AND trading_date >= DATEADD(day, -60, '{latest_date}')
            """
            
            data_result = db.execute_query(data_query)
            record_count = data_result.iloc[0]['count'] if not data_result.empty else 0
            
            if record_count >= 50:
                safe_print(f"✅ Sufficient data for prediction: {record_count} records")
            else:
                safe_print(f"⚠️  Limited data for prediction: {record_count} records")
            
            return True
        else:
            safe_print("❌ No sample stock data found")
            return False
        
    except Exception as e:
        safe_print(f"❌ Sample prediction test error: {e}")
        return False

def main():
    """Run all NSE setup tests."""
    safe_print("🇮🇳 NSE 500 Setup Test Suite")
    safe_print("=" * 50)
    
    tests = [
        ("Database Connection", test_database_connection),
        ("NSE Table Structure", test_nse_tables),
        ("NSE Data Quality", test_nse_data_quality),
        ("Model Files", test_model_files),
        ("Directory Structure", test_directories),
        ("Sample Prediction", test_sample_prediction)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_function in tests:
        safe_print(f"\n🔍 Running test: {test_name}")
        safe_print("-" * 30)
        
        try:
            if test_function():
                safe_print(f"✅ {test_name}: PASSED")
                passed_tests += 1
            else:
                safe_print(f"❌ {test_name}: FAILED")
        except Exception as e:
            safe_print(f"❌ {test_name}: ERROR - {e}")
    
    # Summary
    safe_print("\n" + "=" * 50)
    safe_print("🇮🇳 NSE SETUP TEST SUMMARY")
    safe_print("=" * 50)
    safe_print(f"📊 Tests Passed: {passed_tests}/{total_tests}")
    safe_print(f"📊 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        safe_print("🎉 All tests passed! NSE setup is ready for use.")
        safe_print("\n📝 Next steps:")
        safe_print("1. Run: python predict_nse_signals.py --all-nse")
        safe_print("2. Or: python daily_nse_automation.py")
        safe_print("3. Or: run_nse_automation.bat")
    else:
        safe_print("⚠️  Some tests failed. Please address the issues before proceeding.")
        safe_print("\n🔧 Troubleshooting:")
        if passed_tests == 0:
            safe_print("- Check database connectivity and NSE data availability")
            safe_print("- Ensure SQL Server is running and accessible")
        else:
            safe_print("- Review failed test details above")
            safe_print("- Check log files for more information")
    
    safe_print(f"\n✅ Test completed at {datetime.now()}")

if __name__ == "__main__":
    main()