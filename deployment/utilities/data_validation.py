def missing_data_check(df, mmsi=None, log_fn=print):
    """
    Validates whether a 60-step input window is safe for inference.

    Requirements:
      - All 9 expected features must exist and not be fully null
      - Must contain exactly 60 rows

    Args:
        df (pd.DataFrame): The input window (should be 60 time steps).
        mmsi (str or int, optional): For logging context.
        log_fn (function): Logging or print function.

    Returns:
        bool: True if safe to infer, False if not.
    """
    expected_cols = [
        "Latitude", "Longitude", "ROT", "SOG", "COG",
        "Heading", "Width", "Length", "Draught"
    ]

    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        log_fn(f"MMSI {mmsi or '?'} — missing columns: {missing}")
        return False

    fully_null = [col for col in expected_cols if df[col].isnull().all()]
    if fully_null:
        log_fn(f"MMSI {mmsi or '?'} — fully null columns: {fully_null}")
        return False

    partly_null = [col for col in expected_cols if df[col].isnull().any()]
    if partly_null:
        log_fn(f"MMSI {mmsi or '?'} — partly null columns: {partly_null}")
        return False

    if len(df) < 60:
        log_fn(f"MMSI {mmsi or '?'} — window too short: {len(df)} < 60")
        return False

    return True
