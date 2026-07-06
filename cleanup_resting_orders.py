#!/usr/bin/env python3
"""One-time cleanup script for existing resting GTC orders.

Run this once to sync all resting orders with Kalshi and clean up expired/cancelled ones.
"""

import sys
import time
from app import create_app
from app.models import LiveTrade, db
from app.kalshi_trader import get_order_status
from app.db_helpers import set_setting

def cleanup_resting_orders():
    """Sync all resting orders with Kalshi and update their status."""
    app = create_app()
    
    with app.app_context():
        # Find all resting orders
        resting_orders = LiveTrade.query.filter(
            LiveTrade.order_status == "resting",
            LiveTrade.resolved.is_(False),
        ).all()
        
        if not resting_orders:
            print("✓ No resting orders found - database is clean")
            return
        
        print(f"Found {len(resting_orders)} resting order(s) to check")
        print("-" * 80)
        
        cleaned = 0
        filled = 0
        still_resting = 0
        errors = 0
        
        for i, trade in enumerate(resting_orders, 1):
            print(f"\n[{i}/{len(resting_orders)}] Checking order {trade.kalshi_order_id}")
            print(f"  Ticker: {trade.ticker}")
            print(f"  Side: {trade.side}")
            print(f"  Entry: {trade.entry_at}")
            
            if not trade.kalshi_order_id:
                print("  ⚠ No order ID - marking as failed")
                trade.order_status = "failed"
                trade.error_detail = "No order ID recorded"
                cleaned += 1
                continue
            
            # Query Kalshi for current status
            status = get_order_status(trade.kalshi_order_id)
            if status is None:
                print("  ✗ Could not fetch status from Kalshi")
                errors += 1
                continue
            
            kalshi_status = status.get("status", "unknown")
            fill_count = status.get("fill_count", 0)
            
            print(f"  Kalshi status: {kalshi_status}")
            print(f"  Fill count: {fill_count}")
            
            if fill_count >= 1:
                # Order filled!
                trade.order_status = "placed"
                trade.contracts = fill_count
                avg_price = status.get("average_fill_price")
                if avg_price is not None:
                    trade.entry_price = float(avg_price)
                    trade.entry_price_cents = max(1, min(99, int(round(float(avg_price) * 100))))
                fill_cost = status.get("fill_cost_dollars")
                if fill_cost is not None:
                    trade.cost_dollars = float(fill_cost)
                else:
                    trade.cost_dollars = fill_count * trade.entry_price
                trade.error_detail = None
                set_setting(f"live_resting_order_{trade.ticker}", "")
                print(f"  ✓ FILLED: {fill_count} contracts at ${trade.entry_price:.3f}")
                filled += 1
                
            elif kalshi_status in ("cancelled", "expired", "rejected"):
                # Order was killed
                trade.order_status = kalshi_status
                if kalshi_status == "expired":
                    trade.error_detail = "Order expired before fill"
                elif kalshi_status == "cancelled":
                    trade.error_detail = "Order cancelled externally"
                else:
                    trade.error_detail = f"Order {kalshi_status}"
                set_setting(f"live_resting_order_{trade.ticker}", "")
                print(f"  ✓ Cleaned up: {kalshi_status}")
                cleaned += 1
                
            else:
                # Still resting
                print(f"  → Still resting (status: {kalshi_status})")
                still_resting += 1
            
            # Rate limit protection
            if i < len(resting_orders):
                time.sleep(0.5)
        
        # Commit all changes
        db.session.commit()
        
        print("\n" + "=" * 80)
        print("CLEANUP SUMMARY")
        print("=" * 80)
        print(f"Total orders checked: {len(resting_orders)}")
        print(f"  ✓ Filled: {filled}")
        print(f"  ✓ Cleaned up (expired/cancelled): {cleaned}")
        print(f"  → Still resting: {still_resting}")
        print(f"  ✗ Errors: {errors}")
        print("=" * 80)
        
        if cleaned > 0 or filled > 0:
            print("\n✓ Database updated successfully")
        
        if still_resting > 0:
            print(f"\n⚠ {still_resting} order(s) still resting - they may fill soon")
            print("  The new cleanup job will check them every 5 minutes")

if __name__ == "__main__":
    try:
        cleanup_resting_orders()
    except KeyboardInterrupt:
        print("\n\nCleanup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Cleanup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
