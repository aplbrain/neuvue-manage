import requests
import backoff
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm
@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=3)
def get_latest_roots(cave_client, seg_id):
    new_root = cave_client.chunkedgraph.get_latest_roots(seg_id)[-1]
    return new_root


@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=3)
def execute_split_from_points(cave_client, source_points, sink_points, root_id):
    operation_id, new_root_ids = cave_client.chunkedgraph.execute_split(source_points=source_points,
                                           sink_points=sink_points,
                                           root_id=root_id)
    return operation_id, new_root_ids


@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=3)
def get_leaves(cave_client, root_id, bounds):
    local_sv = cave_client.chunkedgraph.get_leaves(root_id=root_id, bounds=bounds)
    return local_sv


@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=3)
def preview_split(cave_client, source_points, sink_points, root_id):
    source_cc, sink_cc, success = cave_client.chunkedgraph.preview_split(source_points=source_points, sink_points=sink_points,
                                                                         root_id=root_id)

    return source_cc, sink_cc, success


@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=3)
def get_layer2_ids(cave_client, local_cc):
    source_layer_2 = cave_client.chunkedgraph.get_roots(local_cc, stop_layer=2)
    return source_layer_2


def add_coordinate_to_df(df, cave_client, materialization):
    past_timestamp = cave_client.materialize.get_timestamp(materialization)

    seg_ids = df['post_pt_root_id'].tolist()
    pt_position = Parallel(n_jobs=10)(
        delayed(_get_position_from_synapse)(cave_client, past_timestamp, i) 
        for i in tqdm(seg_ids)
        )
    df['pt_position'] = pt_position
    return df

def _get_position_from_synapse(caveclient, past_timestamp, seg_id):
        try:
            post_synapses = get_post_synapse(caveclient, past_timestamp, seg_id)
            return post_synapses['ctr_pt_position'].values[0]
        except:
            return pd.NA

@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=3)
def get_post_synapse(cave_client, past_timestamp, seg_id):
    post_synapses = cave_client.materialize.query_table(
        "synapses_pni_2",
        filter_in_dict={"post_pt_root_id": [seg_id]},
        select_columns=['ctr_pt_position'], limit=1,
        timestamp=past_timestamp
    )
    return post_synapses
