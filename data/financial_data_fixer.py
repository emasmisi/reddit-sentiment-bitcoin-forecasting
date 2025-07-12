#!/usr/bin/env python3
"""
🔧 FINANCIAL DATA FIXER
=======================
Fix per errore "unsupported operand type(s) for -: 'str' and 'str'"
- Identifica colonne problematiche
- Converte stringhe in numeri
- Valida e pulisce dati finanziari
"""

import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime

class FinancialDataFixer:
    """🔧 Fixer per dati finanziari problematici"""
    
    def __init__(self):
        print("🔧 === FINANCIAL DATA FIXER ===")
        print("🎯 Fixing data type issues in financial files\n")
        
    def inspect_financial_files(self):
        """👀 Ispeziona files finanziari per identificare problemi"""
        
        print("👀 Inspecting financial data files...")
        
        financial_files = glob.glob("data/financial/*.csv")
        
        if not financial_files:
            print("❌ No financial files found!")
            return False
            
        problems_found = []
        
        for file_path in financial_files:
            filename = os.path.basename(file_path)
            asset_name = filename.split('_')[0]
            
            print(f"\n📊 Checking {asset_name.upper()}:")
            
            try:
                df = pd.read_csv(file_path)
                
                print(f"   📋 Shape: {df.shape}")
                print(f"   📅 Columns: {list(df.columns)}")
                
                # Check data types
                print(f"   🔍 Data types:")
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in df.columns:
                        dtype = df[col].dtype
                        print(f"      {col}: {dtype}")
                        
                        # Check for string values in numeric columns
                        if dtype == 'object':
                            problems_found.append((asset_name, col, "String values in numeric column"))
                            print(f"      ⚠️ {col} is object type (should be numeric)")
                            
                            # Show sample values
                            sample_values = df[col].head(3).tolist()
                            print(f"      📝 Sample values: {sample_values}")
                            
                # Check for NaN values
                nan_cols = df.isnull().sum()
                if nan_cols.sum() > 0:
                    print(f"   ⚠️ NaN values found:")
                    for col, count in nan_cols[nan_cols > 0].items():
                        print(f"      {col}: {count} NaN values")
                        
            except Exception as e:
                print(f"   ❌ Error reading {filename}: {e}")
                problems_found.append((asset_name, "file_read", str(e)))
                
        print(f"\n📋 SUMMARY:")
        print(f"   📊 Files checked: {len(financial_files)}")
        print(f"   ⚠️ Problems found: {len(problems_found)}")
        
        if problems_found:
            print(f"\n🔧 PROBLEMS DETAILS:")
            for asset, col, problem in problems_found:
                print(f"   {asset}/{col}: {problem}")
                
        return len(problems_found) == 0
        
    def fix_financial_data(self):
        """🔧 Fix data type issues in financial files"""
        
        print("\n🔧 Fixing financial data files...")
        
        financial_files = glob.glob("data/financial/*.csv")
        fixed_files = []
        failed_files = []
        
        for file_path in financial_files:
            filename = os.path.basename(file_path)
            asset_name = filename.split('_')[0]
            
            print(f"\n🔧 Fixing {asset_name.upper()}:")
            
            try:
                df = pd.read_csv(file_path)
                original_shape = df.shape
                
                # Fix datetime column
                if 'Datetime' in df.columns:
                    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
                    
                # Fix numeric columns
                numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
                
                for col in numeric_columns:
                    if col in df.columns:
                        # Convert to numeric, forcing errors to NaN
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        print(f"   ✅ {col}: converted to numeric")
                        
                # Remove rows with all NaN values in key columns
                key_cols = ['Open', 'High', 'Low', 'Close']
                available_key_cols = [col for col in key_cols if col in df.columns]
                
                if available_key_cols:
                    before_dropna = len(df)
                    df = df.dropna(subset=available_key_cols, how='all')
                    after_dropna = len(df)
                    
                    if before_dropna != after_dropna:
                        print(f"   🧹 Removed {before_dropna - after_dropna} rows with missing key data")
                        
                # Calculate derived columns safely
                try:
                    if 'Open' in df.columns and 'Close' in df.columns:
                        df['daily_return'] = (df['Close'] - df['Open']) / df['Open']
                        print(f"   ✅ Calculated daily_return")
                        
                    if 'High' in df.columns and 'Low' in df.columns and 'Open' in df.columns:
                        df['daily_range'] = (df['High'] - df['Low']) / df['Open']
                        print(f"   ✅ Calculated daily_range")
                        
                    if 'Close' in df.columns:
                        df['price_change'] = df['Close'].pct_change()
                        df['volatility_3d'] = df['price_change'].rolling(3).std()
                        df['volatility_7d'] = df['price_change'].rolling(7).std()
                        print(f"   ✅ Calculated volatility metrics")
                        
                except Exception as calc_error:
                    print(f"   ⚠️ Warning in calculations: {calc_error}")
                    
                # Save fixed file
                backup_path = file_path.replace('.csv', '_backup.csv')
                df_original = pd.read_csv(file_path)
                df_original.to_csv(backup_path, index=False)
                
                df.to_csv(file_path, index=False)
                
                print(f"   💾 Fixed data saved (backup: {os.path.basename(backup_path)})")
                print(f"   📊 Shape: {original_shape} → {df.shape}")
                
                fixed_files.append(asset_name)
                
            except Exception as e:
                print(f"   ❌ Failed to fix {asset_name}: {e}")
                failed_files.append((asset_name, str(e)))
                
        print(f"\n✅ FIXING SUMMARY:")
        print(f"   🔧 Files fixed: {len(fixed_files)}")
        print(f"   ❌ Files failed: {len(failed_files)}")
        
        if fixed_files:
            print(f"   ✅ Successfully fixed: {', '.join(fixed_files)}")
            
        if failed_files:
            print(f"   ❌ Failed to fix:")
            for asset, error in failed_files:
                print(f"      {asset}: {error}")
                
        return len(fixed_files) > 0
        
    def validate_fixed_data(self):
        """✅ Valida che i dati siano stati corretti"""
        
        print(f"\n✅ Validating fixed financial data...")
        
        financial_files = glob.glob("data/financial/*.csv")
        validation_results = {}
        
        for file_path in financial_files:
            if '_backup.csv' in file_path:
                continue  # Skip backup files
                
            filename = os.path.basename(file_path)
            asset_name = filename.split('_')[0]
            
            try:
                df = pd.read_csv(file_path)
                
                validation = {
                    'total_rows': len(df),
                    'datetime_ok': False,
                    'numeric_columns_ok': True,
                    'calculations_ok': True,
                    'issues': []
                }
                
                # Check datetime
                if 'Datetime' in df.columns:
                    try:
                        pd.to_datetime(df['Datetime'])
                        validation['datetime_ok'] = True
                    except:
                        validation['issues'].append("Datetime conversion issues")
                        
                # Check numeric columns
                numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in numeric_cols:
                    if col in df.columns:
                        if df[col].dtype not in ['float64', 'int64', 'float32', 'int32']:
                            validation['numeric_columns_ok'] = False
                            validation['issues'].append(f"{col} is not numeric")
                            
                # Test basic calculations
                try:
                    if 'Open' in df.columns and 'Close' in df.columns:
                        test_calc = (df['Close'] - df['Open']) / df['Open']
                        if test_calc.isna().all():
                            validation['calculations_ok'] = False
                            validation['issues'].append("All calculations result in NaN")
                except:
                    validation['calculations_ok'] = False
                    validation['issues'].append("Calculation test failed")
                    
                validation_results[asset_name] = validation
                
                # Print validation results
                status = "✅" if (validation['datetime_ok'] and validation['numeric_columns_ok'] and validation['calculations_ok']) else "⚠️"
                print(f"   {status} {asset_name.upper()}: {validation['total_rows']} rows")
                
                if validation['issues']:
                    for issue in validation['issues']:
                        print(f"      ⚠️ {issue}")
                        
            except Exception as e:
                print(f"   ❌ {asset_name}: Validation failed - {e}")
                validation_results[asset_name] = {'error': str(e)}
                
        # Overall validation status
        successful_validations = sum(1 for v in validation_results.values() 
                                   if 'error' not in v and v.get('numeric_columns_ok', False))
        
        print(f"\n📊 VALIDATION SUMMARY:")
        print(f"   ✅ Successfully validated: {successful_validations} files")
        print(f"   📊 Total files checked: {len(validation_results)}")
        
        return successful_validations > 0, validation_results
        
    def run_complete_fix(self):
        """🚀 Esegue fix completo dei dati finanziari"""
        
        print("🚀 === COMPLETE FINANCIAL DATA FIX ===\n")
        
        # Step 1: Inspect
        print("STEP 1: INSPECTION")
        inspection_ok = self.inspect_financial_files()
        
        if inspection_ok:
            print("\n✅ No obvious problems found, but fixing data types anyway...")
        else:
            print("\n⚠️ Problems found, proceeding with fixes...")
            
        # Step 2: Fix
        print(f"\nSTEP 2: FIXING")
        fix_success = self.fix_financial_data()
        
        if not fix_success:
            print("\n❌ Fixing failed!")
            return False
            
        # Step 3: Validate
        print(f"\nSTEP 3: VALIDATION")
        validation_ok, validation_results = self.validate_fixed_data()
        
        if validation_ok:
            print(f"\n🎉 === FIX COMPLETED SUCCESSFULLY ===")
            print(f"✅ Financial data is now ready for correlation analysis!")
            print(f"🚀 Run the correlation analysis again!")
            return True
        else:
            print(f"\n❌ Validation failed - manual inspection needed")
            return False

def main():
    """🚀 Main execution"""
    
    fixer = FinancialDataFixer()
    
    try:
        success = fixer.run_complete_fix()
        
        if success:
            print(f"\n🎯 === NEXT STEP ===")
            print("Run correlation analysis again:")
            print("python comprehensive_correlation_analysis.py")
        else:
            print(f"\n🔧 === MANUAL FIX NEEDED ===")
            print("Check financial data files manually")
            
    except Exception as e:
        print(f"❌ Fixer failed: {e}")

if __name__ == "__main__":
    main()