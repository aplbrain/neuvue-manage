from tqdm.auto import tqdm

def post_tasks_from_state_and_proofreader_df(neuvue_client, state_df, AUTHOR, NAMESPACE, seg_id_str='pt_root_id'):
    """
    Post tasks from a state df.
    """
    failed = []
    for index, row in tqdm(state_df.iterrows()):
        assignees = row['proofreaders']
        state = row['states']
        metadata = row['metadata']

        try:

            resp = neuvue_client.post_task_broadcast(
                author=AUTHOR,
                assignees=assignees,
                priority=1000,
                namespace=NAMESPACE,
                instructions=row['instructions'],
                seg_id=str(row[seg_id_str]),
                metadata=metadata,
                ng_state=state,
            )
        except:
            print(f"Failed to post task {index}")
            failed.append(index)

    return state_df, failed


def post_tasks_from_state_df_broadcast(neuvue_client, state_df, AUTHOR, ASSIGNEES, NAMESPACE, INSTRUCTIONS):
    """
    Post tasks from a state df.
    """
    if isinstance(ASSIGNEES, str):
        raise ValueError('ASSIGNEES must be a list, not a str')

    for index, row in tqdm(state_df.iterrows()):
            state = row['states']

            metadata = row['metadata']

            resp = neuvue_client.post_task_broadcast(
                author=AUTHOR,
                assignees=ASSIGNEES,
                priority=1000,
                namespace=NAMESPACE,
                instructions=INSTRUCTIONS,
                seg_id=str(row['pt_root_id']),
                metadata=metadata,
                ng_state=state
            )
    return

