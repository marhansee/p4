def sliding_windows(df, window_size=60, step_size=1):
    for start in range(0, len(df) - window_size + 1, step_size):
        yield df.iloc[start:start + window_size]
