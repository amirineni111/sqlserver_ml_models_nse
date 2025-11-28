"""
NSE Trading Signals Export Script

This script exports NSE trading signal predictions, technical indicators, and summaries
to CSV files for analysis and reporting.

Features:
- Exports latest NSE trading predictions
- Exports technical indicators data
- Exports prediction summaries
- Creates timestamped CSV files
- Filters by confidence levels
- Supports date range filtering

Usage:
    python export_nse_results.py --all                    # Export all data
    python export_nse_results.py --predictions            # Export only predictions
    python export_nse_results.py --high-confidence        # Export high confidence only
    python export_nse_results.py --date 2025-11-28        # Export for specific date
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import json

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from database.connection import SQLServerConnection

def safe_print(text):
    """Print text with safe encoding handling."""
    try:
        print(text)
    except UnicodeEncodeError:
        emoji_replacements = {
            '✅': '[SUCCESS]', '❌': '[ERROR]', '📊': '[DATA]',
            '📈': '[EXPORT]', '📁': '[FILE]', '🇮🇳': '[NSE]',
            '📋': '[SUMMARY]', '🎯': '[HIGH-CONF]', '💾': '[SAVE]'
        }
        for emoji, replacement in emoji_replacements.items():
            text = text.replace(emoji, replacement)
        print(text)

class NSEResultsExporter:
    """NSE trading results export utility"""
    
    def __init__(self):
        self.db = SQLServerConnection()
        self.export_dir = "results"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create export directory if it doesn't exist
        os.makedirs(self.export_dir, exist_ok=True)
    
    def export_predictions(self, date_filter=None, confidence_filter=None):
        """Export NSE trading predictions to CSV"""
        try:
            safe_print("📈 Exporting NSE trading predictions...")
            
            # Build query
            query = """
            SELECT 
                trading_date,
                ticker,
                company,
                predicted_signal,
                confidence,
                confidence_percentage,
                signal_strength,
                close_price,
                volume,
                rsi,
                rsi_category,
                high_confidence,
                medium_confidence,
                low_confidence,
                sell_probability,
                buy_probability,
                hold_probability,
                model_name,
                created_at
            FROM ml_nse_trading_predictions
            """
            
            where_conditions = []
            
            if date_filter:
                where_conditions.append(f"trading_date = '{date_filter}'")
            else:
                # Default to last 7 days
                where_conditions.append("trading_date >= DATEADD(day, -7, GETDATE())")
            
            if confidence_filter == 'high':
                where_conditions.append("high_confidence = 1")
            elif confidence_filter == 'medium':
                where_conditions.append("medium_confidence = 1")
            elif confidence_filter == 'low':
                where_conditions.append("low_confidence = 1")
            
            if where_conditions:
                query += " WHERE " + " AND ".join(where_conditions)
            
            query += " ORDER BY trading_date DESC, confidence DESC, ticker"
            
            df = self.db.execute_query(query)
            
            if df.empty:
                safe_print("⚠️  No predictions found for export criteria")
                return None
            
            # Generate filename
            conf_suffix = f"_{confidence_filter}_confidence" if confidence_filter else ""
            date_suffix = f"_{date_filter}" if date_filter else ""
            filename = f"nse_trading_predictions{conf_suffix}{date_suffix}_{self.timestamp}.csv"
            filepath = os.path.join(self.export_dir, filename)
            
            # Export to CSV
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            safe_print(f"📁 Exported {len(df)} predictions to: {filename}")
            
            return filepath
            
        except Exception as e:
            safe_print(f"❌ Error exporting predictions: {e}")
            return None
    
    def export_technical_indicators(self, date_filter=None):
        """Export NSE technical indicators to CSV"""
        try:
            safe_print("📊 Exporting NSE technical indicators...")
            
            query = """
            SELECT 
                trading_date,
                ticker,
                sma_5, sma_10, sma_20, sma_50,
                ema_5, ema_10, ema_20, ema_50,
                macd, macd_signal, macd_histogram, macd_trend,
                rsi, rsi_oversold, rsi_overbought, rsi_momentum,
                bb_upper, bb_middle, bb_lower, bb_width,
                atr, atr_percentage,
                price_vs_sma20, price_vs_sma20_pct,
                price_vs_sma50, price_vs_sma50_pct,
                price_vs_ema20,
                sma20_vs_sma50, ema20_vs_ema50, sma5_vs_sma20,
                trend_direction,
                volume_sma_20, volume_sma_ratio,
                price_momentum_5, price_momentum_10,
                daily_volatility, price_volatility_10, price_volatility_20,
                trend_strength_10, volume_price_trend, gap,
                created_at
            FROM ml_nse_technical_indicators
            """
            
            if date_filter:
                query += f" WHERE trading_date = '{date_filter}'"
            else:
                query += " WHERE trading_date >= DATEADD(day, -7, GETDATE())"
            
            query += " ORDER BY trading_date DESC, ticker"
            
            df = self.db.execute_query(query)
            
            if df.empty:
                safe_print("⚠️  No technical indicators found for export criteria")
                return None
            
            # Generate filename
            date_suffix = f"_{date_filter}" if date_filter else ""
            filename = f"nse_technical_indicators{date_suffix}_{self.timestamp}.csv"
            filepath = os.path.join(self.export_dir, filename)
            
            # Export to CSV
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            safe_print(f"📁 Exported {len(df)} technical indicator records to: {filename}")
            
            return filepath
            
        except Exception as e:
            safe_print(f"❌ Error exporting technical indicators: {e}")
            return None
    
    def export_prediction_summary(self, date_filter=None):
        """Export NSE prediction summaries to CSV"""
        try:
            safe_print("📋 Exporting NSE prediction summaries...")
            
            query = """
            SELECT 
                analysis_date,
                total_predictions,
                total_buy_signals,
                total_sell_signals,
                total_hold_signals,
                high_confidence_count,
                medium_confidence_count,
                low_confidence_count,
                high_conf_buys,
                medium_conf_buys,
                low_conf_buys,
                high_conf_sells,
                medium_conf_sells,
                low_conf_sells,
                avg_confidence,
                avg_rsi,
                avg_buy_probability,
                avg_sell_probability,
                market_trend,
                bullish_stocks_count,
                bearish_stocks_count,
                neutral_stocks_count,
                processing_time_seconds,
                total_stocks_processed,
                failed_predictions,
                data_quality_score,
                created_at,
                notes
            FROM ml_nse_predict_summary
            """
            
            if date_filter:
                query += f" WHERE analysis_date = '{date_filter}'"
            else:
                query += " WHERE analysis_date >= DATEADD(day, -30, GETDATE())"
            
            query += " ORDER BY analysis_date DESC"
            
            df = self.db.execute_query(query)
            
            if df.empty:
                safe_print("⚠️  No prediction summaries found for export criteria")
                return None
            
            # Generate filename
            date_suffix = f"_{date_filter}" if date_filter else ""
            filename = f"nse_prediction_summary{date_suffix}_{self.timestamp}.csv"
            filepath = os.path.join(self.export_dir, filename)
            
            # Export to CSV
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            safe_print(f"📁 Exported {len(df)} summary records to: {filename}")
            
            return filepath
            
        except Exception as e:
            safe_print(f"❌ Error exporting prediction summary: {e}")
            return None
    
    def export_high_confidence_signals(self, date_filter=None, signal_type=None):
        """Export high confidence NSE signals for trading decisions"""
        try:
            safe_print("🎯 Exporting high confidence NSE signals...")
            
            query = """
            SELECT 
                p.trading_date,
                p.ticker,
                p.company,
                p.predicted_signal,
                p.confidence_percentage,
                p.close_price,
                p.volume,
                p.rsi,
                p.buy_probability,
                p.sell_probability,
                -- Technical indicators
                t.sma_20,
                t.sma_50,
                t.ema_20,
                t.macd,
                t.macd_signal,
                t.bb_upper,
                t.bb_lower,
                t.atr_percentage,
                t.trend_direction,
                t.price_vs_sma20_pct,
                t.volume_sma_ratio
            FROM ml_nse_trading_predictions p
            LEFT JOIN ml_nse_technical_indicators t 
                ON p.ticker = t.ticker AND p.trading_date = t.trading_date
            WHERE p.high_confidence = 1
            """
            
            conditions = []
            
            if date_filter:
                conditions.append(f"p.trading_date = '{date_filter}'")
            else:
                conditions.append("p.trading_date >= DATEADD(day, -3, GETDATE())")
            
            if signal_type:
                conditions.append(f"p.predicted_signal = '{signal_type}'")
            
            if conditions:
                query += " AND " + " AND ".join(conditions)
            
            query += " ORDER BY p.confidence_percentage DESC, p.ticker"
            
            df = self.db.execute_query(query)
            
            if df.empty:
                safe_print("⚠️  No high confidence signals found for export criteria")
                return None
            
            # Generate filename
            signal_suffix = f"_{signal_type.lower()}" if signal_type else ""
            date_suffix = f"_{date_filter}" if date_filter else ""
            filename = f"nse_high_confidence_signals{signal_suffix}{date_suffix}_{self.timestamp}.csv"
            filepath = os.path.join(self.export_dir, filename)
            
            # Export to CSV
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            safe_print(f"📁 Exported {len(df)} high confidence signals to: {filename}")
            
            return filepath
            
        except Exception as e:
            safe_print(f"❌ Error exporting high confidence signals: {e}")
            return None
    
    def export_sector_analysis(self, date_filter=None):
        """Export sector-wise analysis of NSE predictions"""
        try:
            safe_print("📊 Exporting NSE sector analysis...")
            
            # First, get all predictions with sector information
            query = """
            SELECT 
                p.trading_date,
                p.ticker,
                p.company,
                p.predicted_signal,
                p.confidence_percentage,
                p.close_price,
                -- Try to get sector from various sources
                COALESCE(p.sector, 'Unknown') as sector
            FROM ml_nse_trading_predictions p
            """
            
            if date_filter:
                query += f" WHERE p.trading_date = '{date_filter}'"
            else:
                query += " WHERE p.trading_date >= DATEADD(day, -1, GETDATE())"
            
            query += " ORDER BY p.trading_date DESC, p.ticker"
            
            df = self.db.execute_query(query)
            
            if df.empty:
                safe_print("⚠️  No data found for sector analysis")
                return None
            
            # Create sector summary
            sector_summary = df.groupby(['sector', 'predicted_signal']).agg({
                'ticker': 'count',
                'confidence_percentage': 'mean',
                'close_price': 'mean'
            }).round(2)
            
            # Reset index to make it more readable
            sector_summary = sector_summary.reset_index()
            sector_summary.columns = ['Sector', 'Signal', 'Count', 'Avg_Confidence', 'Avg_Price']
            
            # Generate filename
            date_suffix = f"_{date_filter}" if date_filter else ""
            filename = f"nse_sector_analysis{date_suffix}_{self.timestamp}.csv"
            filepath = os.path.join(self.export_dir, filename)
            
            # Export to CSV
            sector_summary.to_csv(filepath, index=False, encoding='utf-8-sig')
            safe_print(f"📁 Exported sector analysis to: {filename}")
            
            return filepath
            
        except Exception as e:
            safe_print(f"❌ Error exporting sector analysis: {e}")
            return None
    
    def create_trading_watchlist(self, date_filter=None):
        """Create a focused trading watchlist from NSE predictions"""
        try:
            safe_print("📋 Creating NSE trading watchlist...")
            
            # Get high and medium confidence signals
            query = """
            SELECT TOP 50
                p.trading_date,
                p.ticker,
                p.company,
                p.predicted_signal,
                p.confidence_percentage,
                p.close_price,
                p.volume,
                p.rsi,
                p.buy_probability,
                p.sell_probability,
                -- Key technical indicators
                t.sma_20,
                t.ema_20,
                t.macd_trend,
                t.trend_direction,
                t.price_vs_sma20_pct,
                t.atr_percentage,
                -- Risk indicators
                CASE 
                    WHEN p.rsi > 70 THEN 'Overbought Risk'
                    WHEN p.rsi < 30 THEN 'Oversold Opportunity'
                    ELSE 'Normal'
                END as risk_level,
                -- Action recommendation
                CASE 
                    WHEN p.predicted_signal = 'Buy' AND p.confidence_percentage >= 80 THEN 'Strong Buy'
                    WHEN p.predicted_signal = 'Buy' AND p.confidence_percentage >= 65 THEN 'Buy'
                    WHEN p.predicted_signal = 'Sell' AND p.confidence_percentage >= 80 THEN 'Strong Sell'
                    WHEN p.predicted_signal = 'Sell' AND p.confidence_percentage >= 65 THEN 'Sell'
                    ELSE 'Hold/Watch'
                END as action_recommendation
            FROM ml_nse_trading_predictions p
            LEFT JOIN ml_nse_technical_indicators t 
                ON p.ticker = t.ticker AND p.trading_date = t.trading_date
            WHERE (p.high_confidence = 1 OR p.medium_confidence = 1)
                AND p.predicted_signal IN ('Buy', 'Sell')
            """
            
            if date_filter:
                query += f" AND p.trading_date = '{date_filter}'"
            else:
                query += " AND p.trading_date >= DATEADD(day, -1, GETDATE())"
            
            query += " ORDER BY p.confidence_percentage DESC, p.buy_probability DESC"
            
            df = self.db.execute_query(query)
            
            if df.empty:
                safe_print("⚠️  No watchlist candidates found")
                return None
            
            # Generate filename
            date_suffix = f"_{date_filter}" if date_filter else ""
            filename = f"nse_trading_watchlist{date_suffix}_{self.timestamp}.csv"
            filepath = os.path.join(self.export_dir, filename)
            
            # Export to CSV
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            safe_print(f"📁 Created trading watchlist with {len(df)} stocks: {filename}")
            
            return filepath
            
        except Exception as e:
            safe_print(f"❌ Error creating trading watchlist: {e}")
            return None

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Export NSE Trading Results')
    parser.add_argument('--all', action='store_true', help='Export all data types')
    parser.add_argument('--predictions', action='store_true', help='Export predictions only')
    parser.add_argument('--indicators', action='store_true', help='Export technical indicators only')
    parser.add_argument('--summary', action='store_true', help='Export summaries only')
    parser.add_argument('--watchlist', action='store_true', help='Create trading watchlist')
    parser.add_argument('--sector-analysis', action='store_true', help='Export sector analysis')
    parser.add_argument('--high-confidence', action='store_true', help='Export high confidence signals only')
    parser.add_argument('--date', help='Filter by specific date (YYYY-MM-DD)')
    parser.add_argument('--signal-type', choices=['Buy', 'Sell', 'Hold'], help='Filter by signal type')
    
    args = parser.parse_args()
    
    # Initialize exporter
    exporter = NSEResultsExporter()
    
    safe_print("🇮🇳 Starting NSE Results Export...")
    safe_print(f"📁 Export directory: {exporter.export_dir}")
    
    exported_files = []
    
    try:
        if args.all or args.predictions:
            if args.high_confidence:
                file_path = exporter.export_high_confidence_signals(args.date, args.signal_type)
            else:
                file_path = exporter.export_predictions(args.date, 'high' if args.high_confidence else None)
            if file_path:
                exported_files.append(file_path)
        
        if args.all or args.indicators:
            file_path = exporter.export_technical_indicators(args.date)
            if file_path:
                exported_files.append(file_path)
        
        if args.all or args.summary:
            file_path = exporter.export_prediction_summary(args.date)
            if file_path:
                exported_files.append(file_path)
        
        if args.watchlist:
            file_path = exporter.create_trading_watchlist(args.date)
            if file_path:
                exported_files.append(file_path)
        
        if args.sector_analysis:
            file_path = exporter.export_sector_analysis(args.date)
            if file_path:
                exported_files.append(file_path)
        
        # Default behavior if no specific export type is specified
        if not any([args.all, args.predictions, args.indicators, args.summary, 
                   args.watchlist, args.sector_analysis, args.high_confidence]):
            safe_print("📈 No specific export type specified, exporting predictions...")
            file_path = exporter.export_predictions(args.date)
            if file_path:
                exported_files.append(file_path)
        
        # Summary
        safe_print("\n" + "="*50)
        safe_print("🇮🇳 NSE EXPORT COMPLETED")
        safe_print("="*50)
        safe_print(f"📁 Exported {len(exported_files)} files:")
        for file_path in exported_files:
            safe_print(f"  • {os.path.basename(file_path)}")
        
        if exported_files:
            safe_print(f"\n💾 Files saved in: {exporter.export_dir}")
            safe_print("✅ NSE export completed successfully!")
        else:
            safe_print("⚠️  No files were exported. Check your filters and data availability.")
    
    except Exception as e:
        safe_print(f"❌ Error during export: {e}")

if __name__ == "__main__":
    main()