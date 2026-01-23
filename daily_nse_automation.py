"""
Daily NSE 500 Automation Script

This script provides automated daily workflow for NSE 500 trading signals:
1. Checks NSE data availability and quality
2. Runs NSE predictions for all stocks
3. Saves predictions to database
4. Creates daily reports and summaries
5. Handles errors and logging

Usage:
    python daily_nse_automation.py                    # Full automated daily process
    python daily_nse_automation.py --check-only      # Only check data status
    python daily_nse_automation.py --date 2025-11-28 # Run for specific date
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import json
import time

# Configure UTF-8 encoding for Windows console compatibility
if sys.platform == 'win32':
    # Reconfigure stdout with UTF-8 encoding, ignore errors for Task Scheduler
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        # For older Python versions
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from database.connection import SQLServerConnection

def setup_nse_logging():
    """Setup comprehensive logging for NSE daily automation."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_filename = log_dir / f"daily_nse_automation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    class UnicodeFormatter(logging.Formatter):
        def format(self, record):
            formatted = super().format(record)
            # Replace emoji with text for console output
            emoji_replacements = {
                '[NSE]': '[NSE]', '[INFO]': '[INFO]', '[START]': '[START]', '[LOG]': '[LOG]',
                '[ERROR]': '[ERROR]', '[SUCCESS]': '[SUCCESS]', '[DATA]': '[DATA]', '[PROCESS]': '[PROCESS]',
                '[FILE]': '[FILES]', '[TIME]': '[TIMEOUT]', '[PREDICTION]': '[PREDICTION]', '[DATE]': '[DATE]',
                '[WARN]': '[WARN]', '[COMPLETE]': '[COMPLETE]', '[SUMMARY]': '[SUMMARY]', '[SAVE]': '[SAVE]'
            }
            for emoji, replacement in emoji_replacements.items():
                formatted = formatted.replace(emoji, replacement)
            return formatted
    
    # File handler
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(UnicodeFormatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Setup logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return log_filename

def safe_print(text):
    """Print with safe encoding for Windows Task Scheduler compatibility."""
    # Convert text to string if it's not already
    text = str(text)
    
    # Just in case any emojis slipped through, replace common ones
    emoji_map = {
        '💡': '[TIP]', '🎯': '[TARGET]', '🟢': '[BUY]',
        '🔴': '[SELL]', '🟡': '[MEDIUM]', '⚡': '[FAST]', '🔔': '[NOTIFY]'
    }
    
    for emoji, replacement in emoji_map.items():
        text = text.replace(emoji, replacement)
    
    try:
        print(text)
    except (UnicodeEncodeError, UnicodeDecodeError):
        # If still failing, encode to ascii ignoring errors
        safe_text = text.encode('ascii', 'ignore').decode('ascii')
        print(safe_text)

def check_nse_data_status():
    """Check NSE data availability and quality."""
    try:
        logging.info("[INFO] Checking NSE data status and connectivity...")
        
        db = SQLServerConnection()
        
        # Test database connection
        if not db.test_connection():
            logging.error("[ERROR] Database connection failed")
            return False, None, None, None
        
        logging.info("[SUCCESS] Database connection successful")
        
        # Check NSE historical data
        nse_data_query = """
        SELECT 
            COUNT(*) as total_records,
            MAX(trading_date) as latest_date,
            MIN(trading_date) as earliest_date,
            COUNT(DISTINCT ticker) as unique_tickers,
            COUNT(DISTINCT trading_date) as unique_dates
        FROM dbo.nse_500_hist_data
        """
        
        nse_result = db.execute_query(nse_data_query)
        
        if nse_result.empty:
            logging.error("[ERROR] No NSE historical data found")
            return False, None, None, None
        
        total_records = nse_result.iloc[0]['total_records']
        latest_date = nse_result.iloc[0]['latest_date']
        unique_tickers = nse_result.iloc[0]['unique_tickers']
        unique_dates = nse_result.iloc[0]['unique_dates']
        
        logging.info(f"[DATA] NSE Historical Data Status:")
        logging.info(f"  [DATE] Latest Date: {latest_date}")
        logging.info(f"  [DATA] Total Records: {total_records:,}")
        logging.info(f"  [COMPANY] Unique Tickers: {unique_tickers}")
        logging.info(f"  [DATE] Unique Dates: {unique_dates}")
        
        # Calculate data age
        if latest_date:
            if hasattr(latest_date, 'date'):
                data_age = (datetime.now().date() - latest_date.date()).days
            else:
                data_age = (datetime.now().date() - latest_date).days
            logging.info(f"  [TIME] Data Age: {data_age} days")
        else:
            data_age = None
        
        # Check for recent predictions
        pred_query = """
        SELECT 
            COUNT(*) as prediction_count,
            MAX(trading_date) as latest_prediction_date
        FROM ml_nse_trading_predictions
        WHERE trading_date >= DATEADD(day, -7, GETDATE())
        """
        
        pred_result = db.execute_query(pred_query)
        prediction_count = pred_result.iloc[0]['prediction_count'] if not pred_result.empty else 0
        latest_prediction_date = pred_result.iloc[0]['latest_prediction_date'] if not pred_result.empty else None
        
        logging.info(f"[PREDICTION] Recent NSE Predictions:")
        logging.info(f"  [DATA] Last 7 Days: {prediction_count} predictions")
        logging.info(f"  [DATE] Latest Prediction: {latest_prediction_date}")
        
        return True, data_age, latest_date, total_records
        
    except Exception as e:
        logging.error(f"[ERROR] Error checking NSE data status: {e}")
        return False, None, None, None

def run_nse_predictions(target_date=None):
    """Run NSE trading signal predictions."""
    try:
        logging.info("[START] Running NSE 500 predictions...")
        
        cmd_args = [sys.executable, "predict_nse_signals.py", "--all-nse"]
        
        if target_date:
            cmd_args.extend(["--date", target_date])
        
        logging.info(f"[PROCESS] Executing: {' '.join(cmd_args)}")
        
        result = subprocess.run(
            cmd_args,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minutes timeout
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode == 0:
            logging.info("[SUCCESS] NSE predictions completed successfully")
            # Log some output for visibility
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines[-10:]:  # Last 10 lines
                if line.strip():
                    logging.info(f"[DATA] {line}")
            return True
        else:
            logging.error(f"[ERROR] NSE predictions failed with return code: {result.returncode}")
            if result.stderr:
                logging.error(f"Error output: {result.stderr}")
            return False
        
    except subprocess.TimeoutExpired:
        logging.error("[TIME] NSE predictions timed out after 30 minutes")
        return False
    except Exception as e:
        logging.error(f"[ERROR] Error running NSE predictions: {e}")
        return False



def generate_daily_report(target_date=None):
    """Generate daily NSE analysis report."""
    try:
        logging.info("[LOG] Generating daily NSE report...")
        
        db = SQLServerConnection()
        
        # Get today's summary
        if target_date:
            date_filter = f"'{target_date}'"
        else:
            date_filter = "CAST(GETDATE() AS DATE)"
        
        summary_query = f"""
        SELECT TOP 1
            analysis_date,
            total_predictions,
            total_buy_signals,
            total_sell_signals,
            total_hold_signals,
            high_confidence_count,
            medium_confidence_count,
            low_confidence_count,
            avg_confidence,
            avg_rsi,
            total_stocks_processed,
            processing_time_seconds
        FROM ml_nse_predict_summary
        WHERE analysis_date = {date_filter}
        ORDER BY created_at DESC
        """
        
        summary_result = db.execute_query(summary_query)
        
        # Get high confidence signals
        high_conf_query = f"""
        SELECT TOP 10
            ticker,
            company,
            predicted_signal,
            confidence_percentage,
            close_price,
            rsi
        FROM ml_nse_trading_predictions
        WHERE trading_date = {date_filter}
            AND high_confidence = 1
        ORDER BY confidence_percentage DESC
        """
        
        high_conf_result = db.execute_query(high_conf_query)
        
        # Create report
        report_date = target_date or datetime.now().strftime('%Y-%m-%d')
        report_content = f"""
NSE 500 Daily Trading Report
================================
[DATE] Date: {report_date}
[TIME] Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
        
        if not summary_result.empty:
            row = summary_result.iloc[0]
            report_content += f"""
[DATA] PREDICTION SUMMARY
--------------------
[PREDICTION] Total Predictions: {row['total_predictions']}
🟢 Buy Signals: {row['total_buy_signals']}
🔴 Sell Signals: {row['total_sell_signals']}
🟡 Hold Signals: {row['total_hold_signals']}

🎯 CONFIDENCE BREAKDOWN
-----------------------
🔥 High Confidence: {row['high_confidence_count']}
[DATA] Medium Confidence: {row['medium_confidence_count']}
📉 Low Confidence: {row['low_confidence_count']}

[PREDICTION] ANALYSIS METRICS
-------------------
⭐ Average Confidence: {row['avg_confidence']:.2f}
[DATA] Average RSI: {row.get('avg_rsi', 0):.1f}
[COMPANY] Stocks Processed: {row['total_stocks_processed']}
[TIME] Processing Time: {row.get('processing_time_seconds', 0):.1f} seconds

"""
        
        if not high_conf_result.empty:
            report_content += """
🎯 TOP HIGH CONFIDENCE SIGNALS
-------------------------------
"""
            for _, row in high_conf_result.iterrows():
                signal_emoji = "🟢" if row['predicted_signal'] == 'Buy' else "🔴" if row['predicted_signal'] == 'Sell' else "🟡"
                report_content += f"{signal_emoji} {row['ticker']}: {row['predicted_signal']} ({row['confidence_percentage']:.1f}%) - ₹{row['close_price']:.2f}\n"
        
        # Save report
        reports_dir = Path("daily_reports")
        reports_dir.mkdir(exist_ok=True)
        
        report_filename = reports_dir / f"nse_daily_report_{report_date.replace('-', '')}.txt"
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # Also save as JSON for programmatic use
        json_report = {
            'date': report_date,
            'generated_at': datetime.now().isoformat(),
            'summary': summary_result.to_dict('records')[0] if not summary_result.empty else {},
            'high_confidence_signals': high_conf_result.to_dict('records') if not high_conf_result.empty else []
        }
        
        json_filename = reports_dir / f"nse_daily_summary_{report_date.replace('-', '')}.json"
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2, default=str)
        
        logging.info(f"[LOG] Daily report saved: {report_filename}")
        logging.info(f"[DATA] JSON summary saved: {json_filename}")
        
        return True
        
    except Exception as e:
        logging.error(f"[ERROR] Error generating daily report: {e}")
        return False

def main():
    """Main automation function."""
    parser = argparse.ArgumentParser(description='Daily NSE 500 Automation')
    parser.add_argument('--check-only', action='store_true', help='Only check data status')
    parser.add_argument('--date', help='Target date for analysis (YYYY-MM-DD)')
    parser.add_argument('--skip-predictions', action='store_true', help='Skip prediction generation')
    
    args = parser.parse_args()
    
    # Setup logging
    log_filename = setup_nse_logging()
    
    start_time = datetime.now()
    
    safe_print("[START] Starting Daily NSE 500 Automation")
    safe_print(f"[DATE] Target Date: {args.date or 'Today'}")
    safe_print(f"[LOG] Log File: {log_filename}")
    
    success_count = 0
    total_steps = 0
    
    try:
        # Step 1: Check data status
        total_steps += 1
        logging.info("=" * 60)
        logging.info("[INFO] STEP 1: Checking NSE Data Status")
        logging.info("=" * 60)
        
        is_connected, data_age, latest_date, total_records = check_nse_data_status()
        
        if not is_connected:
            logging.error("[ERROR] Cannot proceed without database connectivity")
            return
        
        success_count += 1
        
        if args.check_only:
            logging.info("[SUCCESS] Data status check completed")
            return
        
        # Step 2: Run predictions (unless skipped)
        if not args.skip_predictions:
            total_steps += 1
            logging.info("=" * 60)
            logging.info("[PREDICTION] STEP 2: Running NSE Predictions")
            logging.info("=" * 60)
            
            prediction_success = run_nse_predictions(args.date)
            if prediction_success:
                success_count += 1
                logging.info("[SUCCESS] NSE predictions completed successfully (saved to database)")
            else:
                logging.error("[ERROR] NSE predictions failed")
        
        # Step 3: Generate daily report
        total_steps += 1
        logging.info("=" * 60)
        logging.info("[LOG] STEP 3: Generating Daily Report")
        logging.info("=" * 60)
        
        report_success = generate_daily_report(args.date)
        if report_success:
            success_count += 1
            logging.info("[SUCCESS] Daily report generated successfully")
        else:
            logging.error("[ERROR] Daily report generation failed")
        
        # Final summary
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        logging.info("=" * 60)
        logging.info("[COMPLETE] NSE AUTOMATION SUMMARY")
        logging.info("=" * 60)
        logging.info(f"[TIME] Total Processing Time: {processing_time:.1f} seconds")
        logging.info(f"[SUCCESS] Successful Steps: {success_count}/{total_steps}")
        logging.info(f"[DATE] Date Processed: {args.date or datetime.now().strftime('%Y-%m-%d')}")
        
        if success_count == total_steps:
            safe_print("[COMPLETE] NSE automation completed successfully!")
            logging.info("[COMPLETE] All automation steps completed successfully")
        else:
            safe_print(f"[WARN] NSE automation completed with {total_steps - success_count} failures")
            logging.warning(f"Automation completed with {total_steps - success_count} failed steps")
    
    except KeyboardInterrupt:
        logging.info("[STOP] NSE automation interrupted by user")
        safe_print("[STOP] Process interrupted by user")
    except Exception as e:
        logging.error(f"[ERROR] Fatal error in NSE automation: {e}")
        safe_print(f"[ERROR] Fatal error: {e}")

if __name__ == "__main__":
    main()