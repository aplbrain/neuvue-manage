from neuvueclient import NeuvueQueue
from neuvue_queue_task_assignment.neuvue_constants import NEUVUE_QUEUE_URL
from joblib import Parallel, delayed


NAMESPACE = "micronsTest"
ASSIGNEES = ["unassigned_novice"]

if __name__ = "__main__":
    client = NeuvueQueue(NEUVUE_QUEUE_URL)
    tasks = client.get_tasks(sieve={'namespace': NAMESPACE, 'assignee': ASSIGNEES}, select=['_id'])
    for _id in tasks.index:
        client.delete_task(_id)