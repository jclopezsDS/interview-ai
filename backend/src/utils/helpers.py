"""
Helper utilities for text processing, file operations, and logging.

NOTE: Experimental functions currently used in notebooks for research and reporting.
Potential future integration into production API.

This module contains utility functions used across the application
for common operations and data processing tasks.
"""

import time
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, Any
import threading


def implement_rate_limiter(
    requests_per_minute: int = 30,
    requests_per_hour: int = 500,
    burst_allowance: int = 10,
    enable_user_tracking: bool = True,
    enable_ip_tracking: bool = True,
    max_violation_log_size: int = 1000
    ) -> Dict[str, Any]:
    """Create rate limiting with configurable limits per user/session.
    
    Args:
        requests_per_minute: Maximum requests allowed per minute
        requests_per_hour: Maximum requests allowed per hour
        burst_allowance: Additional requests allowed in burst scenarios
        enable_user_tracking: Enable per-user rate limiting
        enable_ip_tracking: Enable per-IP rate limiting
        max_violation_log_size: Maximum number of violations to log per identifier
        
    Returns:
        Dict[str, Any]: Rate limiter system with tracking and enforcement functions
    """
    print(f"[INIT] Creating rate limiter - {requests_per_minute}/min, {requests_per_hour}/hour")
    print(f"[CONFIG] Burst allowance: {burst_allowance}, User tracking: {enable_user_tracking}, IP tracking: {enable_ip_tracking}")
    
    user_requests = defaultdict(lambda: {"minute": deque(), "hour": deque(), "burst_used": 0})
    ip_requests = defaultdict(lambda: {"minute": deque(), "hour": deque(), "burst_used": 0})
    
    violation_log = defaultdict(deque)
    blocked_requests = defaultdict(int)
    
    lock = threading.Lock()
    
    active_trackers = {
        "users": defaultdict(float),
        "ips": defaultdict(float)
    }
    
    def clean_old_requests(request_queue: deque, time_window: int) -> None:
        """Remove requests older than time_window seconds."""
        current_time = time.time()
        while request_queue and current_time - request_queue[0] > time_window:
            request_queue.popleft()
    
    def prune_violation_logs() -> None:
        """Prune violation logs to prevent memory leaks."""
        for identifier in list(violation_log.keys()):
            if len(violation_log[identifier]) > max_violation_log_size:
                while len(violation_log[identifier]) > max_violation_log_size:
                    violation_log[identifier].popleft()
    
    def check_rate_limit(
        identifier: str,
        identifier_type: str = "user",
        request_size: int = 1
    ) -> Dict[str, Any]:
        """Check if request is within rate limits for given identifier."""
        with lock:
            current_time = time.time()
            current_datetime = datetime.now()
            
            print(f"[CHECK] Rate limit for {identifier_type}: {identifier} (size: {request_size})")
            
            if identifier_type == "user" and enable_user_tracking:
                requests_data = user_requests[identifier]
                active_trackers["users"][identifier] = current_time
            elif identifier_type == "ip" and enable_ip_tracking:
                requests_data = ip_requests[identifier]
                active_trackers["ips"][identifier] = current_time
            else:
                print(f"[SKIP] {identifier_type} tracking disabled")
                return {
                    "allowed": True,
                    "identifier": identifier,
                    "identifier_type": identifier_type,
                    "reason": f"{identifier_type} tracking disabled"
                }
            
            clean_old_requests(requests_data["minute"], 60)
            clean_old_requests(requests_data["hour"], 3600)
            
            minute_count = len(requests_data["minute"])
            hour_count = len(requests_data["hour"])
            burst_used = requests_data["burst_used"]
            
            if requests_data["hour"] and current_time - requests_data["hour"][0] > 3600:
                requests_data["burst_used"] = 0
                burst_used = 0
            
            print(f"[USAGE] Current: {minute_count}/min, {hour_count}/hour, burst: {burst_used}")
            
            minute_limit_exceeded = minute_count + request_size > requests_per_minute
            hour_limit_exceeded = hour_count + request_size > requests_per_hour
            
            can_use_burst = burst_used + request_size <= burst_allowance
            
            if minute_limit_exceeded:
                if can_use_burst and not hour_limit_exceeded:
                    for _ in range(request_size):
                        requests_data["minute"].append(current_time)
                        requests_data["hour"].append(current_time)
                    requests_data["burst_used"] += request_size
                    
                    print(f"[ALLOWED] Request allowed using burst ({request_size} burst tokens used)")
                    return {
                        "allowed": True,
                        "identifier": identifier,
                        "identifier_type": identifier_type,
                        "reason": "burst_allowance_used",
                        "burst_tokens_used": request_size,
                        "remaining_burst": burst_allowance - requests_data["burst_used"],
                        "minute_usage": len(requests_data["minute"]),
                        "hour_usage": len(requests_data["hour"]),
                        "reset_time_minute": current_datetime + timedelta(seconds=60),
                        "reset_time_hour": current_datetime + timedelta(seconds=3600)
                    }
                else:
                    violation_log[identifier].append(current_time)
                    if len(violation_log[identifier]) > max_violation_log_size:
                        violation_log[identifier].popleft()
                    blocked_requests[identifier] += 1
                    
                    reset_time = 60 - (current_time - (requests_data["minute"][0] if requests_data["minute"] else current_time))
                    
                    print(f"[BLOCKED] Rate limit exceeded - Next reset in {reset_time:.1f}s")
                    return {
                        "allowed": False,
                        "identifier": identifier,
                        "identifier_type": identifier_type,
                        "reason": "minute_limit_exceeded",
                        "current_usage": minute_count,
                        "limit": requests_per_minute,
                        "retry_after_seconds": max(1, int(reset_time)),
                        "total_violations": len(violation_log[identifier]),
                        "total_blocked": blocked_requests[identifier]
                    }
            
            elif hour_limit_exceeded:
                violation_log[identifier].append(current_time)
                if len(violation_log[identifier]) > max_violation_log_size:
                    violation_log[identifier].popleft()
                blocked_requests[identifier] += 1
                
                reset_time = 3600 - (current_time - (requests_data["hour"][0] if requests_data["hour"] else current_time))
                
                print(f"[BLOCKED] Hour limit exceeded - Next reset in {reset_time/60:.1f}min")
                return {
                    "allowed": False,
                    "identifier": identifier,
                    "identifier_type": identifier_type,
                    "reason": "hour_limit_exceeded",
                    "current_usage": hour_count,
                    "limit": requests_per_hour,
                    "retry_after_seconds": max(60, int(reset_time)),
                    "total_violations": len(violation_log[identifier]),
                    "total_blocked": blocked_requests[identifier]
                }
            
            else:
                for _ in range(request_size):
                    requests_data["minute"].append(current_time)
                    requests_data["hour"].append(current_time)
                
                print(f"[ALLOWED] Request processed successfully")
                return {
                    "allowed": True,
                    "identifier": identifier,
                    "identifier_type": identifier_type,
                    "reason": "within_limits",
                    "minute_usage": len(requests_data["minute"]),
                    "hour_usage": len(requests_data["hour"]),
                    "remaining_minute": requests_per_minute - len(requests_data["minute"]),
                    "remaining_hour": requests_per_hour - len(requests_data["hour"]),
                    "burst_available": burst_allowance - burst_used
                }
    
    def get_usage_stats(identifier: str, identifier_type: str = "user") -> Dict[str, Any]:
        """Get detailed usage statistics for an identifier."""
        with lock:
            if identifier_type == "user" and enable_user_tracking:
                requests_data = user_requests[identifier]
            elif identifier_type == "ip" and enable_ip_tracking:
                requests_data = ip_requests[identifier]
            else:
                return {"error": f"{identifier_type} tracking not enabled"}
            
            current_time = time.time()
            
            clean_old_requests(requests_data["minute"], 60)
            clean_old_requests(requests_data["hour"], 3600)
            
            minute_usage = len(requests_data["minute"])
            hour_usage = len(requests_data["hour"])
            burst_used = requests_data["burst_used"]
            
            return {
                "identifier": identifier,
                "identifier_type": identifier_type,
                "minute_usage": minute_usage,
                "minute_limit": requests_per_minute,
                "minute_remaining": requests_per_minute - minute_usage,
                "hour_usage": hour_usage,
                "hour_limit": requests_per_hour,
                "hour_remaining": requests_per_hour - hour_usage,
                "burst_used": burst_used,
                "burst_available": burst_allowance - burst_used,
                "total_violations": len(violation_log[identifier]),
                "total_blocked": blocked_requests[identifier],
                "last_request_time": requests_data["minute"][-1] if requests_data["minute"] else None
            }
    
    def reset_user_limits(identifier: str, identifier_type: str = "user") -> Dict[str, Any]:
        """Reset rate limits for a specific identifier (admin function)."""
        with lock:
            print(f"[ADMIN] Resetting limits for {identifier_type}: {identifier}")
            
            if identifier_type == "user" and enable_user_tracking:
                if identifier in user_requests:
                    del user_requests[identifier]
                if identifier in active_trackers["users"]:
                    del active_trackers["users"][identifier]
            elif identifier_type == "ip" and enable_ip_tracking:
                if identifier in ip_requests:
                    del ip_requests[identifier]
                if identifier in active_trackers["ips"]:
                    del active_trackers["ips"][identifier]
            
            if identifier in violation_log:
                del violation_log[identifier]
            if identifier in blocked_requests:
                del blocked_requests[identifier]
            
            print(f"[ADMIN] Limits reset successfully for {identifier}")
            return {"success": True, "identifier": identifier, "identifier_type": identifier_type}
    
    def get_system_stats() -> Dict[str, Any]:
        """Get overall system rate limiting statistics."""
        with lock:
            prune_violation_logs()
            
            total_users = len(user_requests) if enable_user_tracking else 0
            total_ips = len(ip_requests) if enable_ip_tracking else 0
            total_violations = sum(len(violations) for violations in violation_log.values())
            total_blocked = sum(blocked_requests.values())
            
            active_users = 0
            active_ips = 0
            current_time = time.time()
            
            if enable_user_tracking:
                for last_active in active_trackers["users"].values():
                    if current_time - last_active < 300:
                        active_users += 1
            
            if enable_ip_tracking:
                for last_active in active_trackers["ips"].values():
                    if current_time - last_active < 300:
                        active_ips += 1
            
            return {
                "total_tracked_users": total_users,
                "total_tracked_ips": total_ips,
                "active_users_5min": active_users,
                "active_ips_5min": active_ips,
                "total_violations": total_violations,
                "total_blocked_requests": total_blocked,
                "rate_limits": {
                    "requests_per_minute": requests_per_minute,
                    "requests_per_hour": requests_per_hour,
                    "burst_allowance": burst_allowance
                },
                "tracking_enabled": {
                    "users": enable_user_tracking,
                    "ips": enable_ip_tracking
                }
            }
    
    rate_limiter_config = {
        "requests_per_minute": requests_per_minute,
        "requests_per_hour": requests_per_hour,
        "burst_allowance": burst_allowance,
        "user_tracking": enable_user_tracking,
        "ip_tracking": enable_ip_tracking,
        "max_violation_log_size": max_violation_log_size,
        "check_limit": check_rate_limit,
        "get_usage": get_usage_stats,
        "reset_limits": reset_user_limits,
        "get_system_stats": get_system_stats,
        "internal_storage": {
            "user_requests": user_requests,
            "ip_requests": ip_requests,
            "violation_log": violation_log,
            "blocked_requests": blocked_requests,
            "active_trackers": active_trackers
        }
    }
    
    print("[COMPLETED] Rate limiter created successfully")
    return rate_limiter_config
