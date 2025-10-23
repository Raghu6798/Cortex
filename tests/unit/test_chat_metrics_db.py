# """
# Test script to programmatically inspect the chat metrics table content.
# This script allows you to query, analyze, and inspect the ChatMetricsDB table.
# """

# import sys
# import os
# from datetime import datetime, timedelta
# from sqlalchemy import func, desc, asc
# from sqlalchemy.orm import Session

# # Add the backend directory to the Python path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from app.db.database import get_db, engine
# from app.db.models import ChatMetricsDB, ChatSessionDB, LLMProviderDB, LLMModelDB
# from app.utils.logger import logger

# def inspect_chat_metrics():
#     """Main function to inspect chat metrics table content."""
#     print("🔍 Chat Metrics Database Inspector")
#     print("=" * 50)
    
#     # Get database session
#     db = next(get_db())
    
#     try:
#         # 1. Basic table information
#         print("\n📊 BASIC TABLE INFORMATION")
#         print("-" * 30)
        
#         total_metrics = db.query(ChatMetricsDB).count()
#         print(f"Total metrics records: {total_metrics}")
        
#         if total_metrics == 0:
#             print("❌ No metrics found in the database.")
#             return
        
#         # 2. Recent metrics (last 10)
#         print("\n🕒 RECENT METRICS (Last 10)")
#         print("-" * 30)
        
#         recent_metrics = db.query(ChatMetricsDB).order_by(desc(ChatMetricsDB.created_at)).limit(10).all()
        
#         for i, metric in enumerate(recent_metrics, 1):
#             print(f"\n{i}. Session: {metric.session_id[:8]}...")
#             print(f"   Provider: {metric.provider_id}")
#             print(f"   Model: {metric.model_id}")
#             print(f"   Tokens: {metric.total_tokens} (Input: {metric.input_tokens}, Output: {metric.output_tokens})")
#             print(f"   Time: {metric.total_time}s (Completion: {metric.completion_time}s)")
#             print(f"   Created: {metric.created_at}")
#             print(f"   Model Name: {metric.model_name}")
#             print(f"   Service Tier: {metric.service_tier}")
        
#         # 3. Provider statistics
#         print("\n🏢 PROVIDER STATISTICS")
#         print("-" * 30)
        
#         provider_stats = db.query(
#             ChatMetricsDB.provider_id,
#             func.count(ChatMetricsDB.id).label('count'),
#             func.sum(ChatMetricsDB.total_tokens).label('total_tokens'),
#             func.avg(ChatMetricsDB.total_tokens).label('avg_tokens'),
#             func.sum(ChatMetricsDB.total_time).label('total_time'),
#             func.avg(ChatMetricsDB.total_time).label('avg_time')
#         ).group_by(ChatMetricsDB.provider_id).all()
        
#         for stat in provider_stats:
#             print(f"\nProvider: {stat.provider_id}")
#             print(f"  Requests: {stat.count}")
#             print(f"  Total Tokens: {stat.total_tokens}")
#             print(f"  Avg Tokens: {stat.avg_tokens:.2f}")
#             print(f"  Total Time: {stat.total_time:.2f}s")
#             print(f"  Avg Time: {stat.avg_time:.2f}s")
        
#         # 4. Model statistics
#         print("\n🤖 MODEL STATISTICS")
#         print("-" * 30)
        
#         model_stats = db.query(
#             ChatMetricsDB.model_id,
#             func.count(ChatMetricsDB.id).label('count'),
#             func.sum(ChatMetricsDB.total_tokens).label('total_tokens'),
#             func.avg(ChatMetricsDB.total_tokens).label('avg_tokens')
#         ).group_by(ChatMetricsDB.model_id).all()
        
#         for stat in model_stats:
#             print(f"\nModel: {stat.model_id}")
#             print(f"  Requests: {stat.count}")
#             print(f"  Total Tokens: {stat.total_tokens}")
#             print(f"  Avg Tokens: {stat.avg_tokens:.2f}")
        
#         # 5. User statistics
#         print("\n👤 USER STATISTICS")
#         print("-" * 30)
        
#         user_stats = db.query(
#             ChatMetricsDB.user_id,
#             func.count(ChatMetricsDB.id).label('count'),
#             func.sum(ChatMetricsDB.total_tokens).label('total_tokens'),
#             func.sum(ChatMetricsDB.total_time).label('total_time')
#         ).group_by(ChatMetricsDB.user_id).all()
        
#         for stat in user_stats:
#             print(f"\nUser: {stat.user_id[:8]}...")
#             print(f"  Requests: {stat.count}")
#             print(f"  Total Tokens: {stat.total_tokens}")
#             print(f"  Total Time: {stat.total_time:.2f}s")
        
#         # 6. Time-based analysis (last 24 hours)
#         print("\n⏰ TIME-BASED ANALYSIS (Last 24 Hours)")
#         print("-" * 30)
        
#         yesterday = datetime.utcnow() - timedelta(days=1)
#         recent_24h = db.query(ChatMetricsDB).filter(
#             ChatMetricsDB.created_at >= yesterday
#         ).all()
        
#         if recent_24h:
#             total_tokens_24h = sum(m.total_tokens or 0 for m in recent_24h)
#             total_time_24h = sum(m.total_time or 0 for m in recent_24h)
#             avg_tokens_24h = total_tokens_24h / len(recent_24h) if recent_24h else 0
#             avg_time_24h = total_time_24h / len(recent_24h) if recent_24h else 0
            
#             print(f"Requests in last 24h: {len(recent_24h)}")
#             print(f"Total tokens: {total_tokens_24h}")
#             print(f"Total time: {total_time_24h:.2f}s")
#             print(f"Avg tokens per request: {avg_tokens_24h:.2f}")
#             print(f"Avg time per request: {avg_time_24h:.2f}s")
#         else:
#             print("No requests in the last 24 hours")
        
#         # 7. Detailed inspection of a specific record
#         print("\n🔬 DETAILED RECORD INSPECTION")
#         print("-" * 30)
        
#         latest_metric = db.query(ChatMetricsDB).order_by(desc(ChatMetricsDB.created_at)).first()
#         if latest_metric:
#             print(f"\nLatest Record Details:")
#             print(f"  ID: {latest_metric.id}")
#             print(f"  Session ID: {latest_metric.session_id}")
#             print(f"  User ID: {latest_metric.user_id}")
#             print(f"  Provider: {latest_metric.provider_id}")
#             print(f"  Model: {latest_metric.model_id}")
#             print(f"  Input Tokens: {latest_metric.input_tokens}")
#             print(f"  Output Tokens: {latest_metric.output_tokens}")
#             print(f"  Total Tokens: {latest_metric.total_tokens}")
#             print(f"  Completion Time: {latest_metric.completion_time}")
#             print(f"  Prompt Time: {latest_metric.prompt_time}")
#             print(f"  Queue Time: {latest_metric.queue_time}")
#             print(f"  Total Time: {latest_metric.total_time}")
#             print(f"  Model Name: {latest_metric.model_name}")
#             print(f"  System Fingerprint: {latest_metric.system_fingerprint}")
#             print(f"  Service Tier: {latest_metric.service_tier}")
#             print(f"  Finish Reason: {latest_metric.finish_reason}")
#             print(f"  Created At: {latest_metric.created_at}")
            
#             if latest_metric.response_metadata:
#                 print(f"\n  Response Metadata:")
#                 for key, value in latest_metric.response_metadata.items():
#                     print(f"    {key}: {value}")
        
#         # 8. Error analysis
#         print("\n❌ ERROR ANALYSIS")
#         print("-" * 30)
        
#         error_records = db.query(ChatMetricsDB).filter(
#             ChatMetricsDB.finish_reason != 'stop'
#         ).all()
        
#         if error_records:
#             print(f"Records with non-stop finish reasons: {len(error_records)}")
#             for record in error_records:
#                 print(f"  Session: {record.session_id[:8]}... | Reason: {record.finish_reason}")
#         else:
#             print("No error records found (all finished with 'stop')")
        
#         # 9. Performance analysis
#         print("\n⚡ PERFORMANCE ANALYSIS")
#         print("-" * 30)
        
#         slow_requests = db.query(ChatMetricsDB).filter(
#             ChatMetricsDB.total_time > 5.0  # More than 5 seconds
#         ).order_by(desc(ChatMetricsDB.total_time)).limit(5).all()
        
#         if slow_requests:
#             print("Slowest requests (>5s):")
#             for req in slow_requests:
#                 print(f"  {req.total_time:.2f}s | {req.provider_id} | {req.model_id}")
#         else:
#             print("No slow requests found")
        
#         fast_requests = db.query(ChatMetricsDB).filter(
#             ChatMetricsDB.total_time < 1.0  # Less than 1 second
#         ).count()
        
#         print(f"Fast requests (<1s): {fast_requests}")
        
#     except Exception as e:
#         logger.error(f"Error inspecting chat metrics: {str(e)}")
#         print(f"❌ Error: {str(e)}")
    
#     finally:
#         db.close()

# def export_metrics_to_csv():
#     """Export metrics to CSV for external analysis."""
#     print("\n📁 EXPORTING METRICS TO CSV")
#     print("-" * 30)
    
#     db = next(get_db())
    
#     try:
#         import csv
        
#         metrics = db.query(ChatMetricsDB).all()
        
#         if not metrics:
#             print("No metrics to export")
#             return
        
#         filename = f"chat_metrics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
#         with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
#             fieldnames = [
#                 'id', 'session_id', 'user_id', 'provider_id', 'model_id',
#                 'input_tokens', 'output_tokens', 'total_tokens',
#                 'completion_time', 'prompt_time', 'queue_time', 'total_time',
#                 'model_name', 'system_fingerprint', 'service_tier', 'finish_reason',
#                 'created_at'
#             ]
            
#             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#             writer.writeheader()
            
#             for metric in metrics:
#                 writer.writerow({
#                     'id': metric.id,
#                     'session_id': metric.session_id,
#                     'user_id': metric.user_id,
#                     'provider_id': metric.provider_id,
#                     'model_id': metric.model_id,
#                     'input_tokens': metric.input_tokens,
#                     'output_tokens': metric.output_tokens,
#                     'total_tokens': metric.total_tokens,
#                     'completion_time': metric.completion_time,
#                     'prompt_time': metric.prompt_time,
#                     'queue_time': metric.queue_time,
#                     'total_time': metric.total_time,
#                     'model_name': metric.model_name,
#                     'system_fingerprint': metric.system_fingerprint,
#                     'service_tier': metric.service_tier,
#                     'finish_reason': metric.finish_reason,
#                     'created_at': metric.created_at
#                 })
        
#         print(f"✅ Exported {len(metrics)} records to {filename}")
        
#     except Exception as e:
#         print(f"❌ Export error: {str(e)}")
    
#     finally:
#         db.close()

# def cleanup_old_metrics(days=30):
#     """Clean up metrics older than specified days."""
#     print(f"\n🧹 CLEANING UP METRICS OLDER THAN {days} DAYS")
#     print("-" * 30)
    
#     db = next(get_db())
    
#     try:
#         cutoff_date = datetime.utcnow() - timedelta(days=days)
        
#         old_metrics = db.query(ChatMetricsDB).filter(
#             ChatMetricsDB.created_at < cutoff_date
#         ).all()
        
#         if old_metrics:
#             print(f"Found {len(old_metrics)} old records to delete")
            
#             # Uncomment the next line to actually delete
#             # db.query(ChatMetricsDB).filter(ChatMetricsDB.created_at < cutoff_date).delete()
#             # db.commit()
            
#             print("⚠️  Deletion commented out for safety. Uncomment to actually delete.")
#         else:
#             print("No old records found")
            
#     except Exception as e:
#         print(f"❌ Cleanup error: {str(e)}")
    
#     finally:
#         db.close()

# if __name__ == "__main__":
#     print("🚀 Starting Chat Metrics Database Inspector...")
    
#     # Run the main inspection
#     inspect_chat_metrics()
    
#     # Ask user if they want to export
#     export_choice = input("\n📁 Do you want to export metrics to CSV? (y/n): ").lower().strip()
#     if export_choice == 'y':
#         export_metrics_to_csv()
    
#     # Ask user if they want to cleanup old records
#     cleanup_choice = input("\n🧹 Do you want to see cleanup options? (y/n): ").lower().strip()
#     if cleanup_choice == 'y':
#         days = input("Enter days to keep (default 30): ").strip()
#         days = int(days) if days.isdigit() else 30
#         cleanup_old_metrics(days)
    
#     print("\n✅ Inspection complete!")


 