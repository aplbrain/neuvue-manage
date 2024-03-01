from tqdm.auto import tqdm

def post_tasks_from_state_and_proofreader_df(neuvue_client, state_df, author, namespace, seg_id_str='pt_root_id'):
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
                author=author,
                assignees=assignees,
                priority=1000,
                namespace=namespace,
                instructions=row['instructions'],
                seg_id=str(row[seg_id_str]),
                metadata=metadata,
                ng_state=state,
            )
        except:
            print(f"Failed to post task {index}")
            failed.append(index)

    return state_df, failed


