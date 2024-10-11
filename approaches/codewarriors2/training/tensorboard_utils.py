import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator



def tensorboard_logs_to_dataframe(log_dir: str) -> pd.DataFrame:
    """
    Function to convert events.out.tfevent files to pandas Dataframe

    Parameters
    ----------
    log_dir: str
        directory containing events.out.tfevent file

    Returns
    -------
    dfsc: pd.DataFrame
        DataFrame with wall_time, step and value columns
    """

    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    tags = event_acc.Tags()['scalars']
    data = {}

    for tag in tags:
        wall_time = []
        step = []
        value = []
        for scalar_event in event_acc.Scalars(tag):
            wall_time.append(scalar_event.wall_time)
            step.append(scalar_event.step)
            value.append(scalar_event.value)
            
        df = pd.DataFrame({'wall_time': wall_time, 'step': step, 'value': value})
        return df
   


