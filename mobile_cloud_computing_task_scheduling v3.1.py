'''
Shane Hussey
EECE7205 Project 2

'''

import time
from copy import deepcopy

class ApplicationError(Exception):
    pass

class Application():
    '''
    acyclic graph data structure with light functionality
    '''

    def __init__(self, task_ids: list,
                 local_execution_times: list,
                 cloud_execution_times: list,
                 parents: list,
                 children: list,
                 T_max: int,
                 energy_usage: list):
        # inputs
        self.task_ids = task_ids
        self.local_execution_times = local_execution_times
        self.cloud_execution_times = cloud_execution_times
        self.parents = parents
        self.children = children
        self.T_max = T_max
        self.energy_usage = energy_usage

        # properties
        self.tasks = []
        self._initialize_tasks()
        self._find_entry_exit_tasks()
        self.sorted_priority_tasks = None
        self.seqeuence_init = None
        self.tasks_final = []


    def _initialize_tasks(self):
        '''
        function to intialize task objects with parent child relationships
        and inherent properties.
        '''

        string_base = "task"
        for t in range(0, len(self.task_ids)):
            local_speeds = [sublist[t] for sublist in self.local_execution_times]
            task_instance = Task(task_id=self.task_ids[t],
                                 parents=self.parents[t],
                                 children=self.children[t],
                                 cloud_speed=self.cloud_execution_times,
                                 local_core_speeds=local_speeds)

            variable_name = f"{string_base}{self.task_ids[t]}"
            setattr(self,variable_name, task_instance)
            self.tasks.append(task_instance)

        # set the parents and children objects
        for t in range(0,len(self.tasks)):
            self.tasks[t]._set_parents(self.tasks)
            self.tasks[t]._set_children(self.tasks)


    def _find_entry_exit_tasks(self):
        '''
        function which gathers entry and exit tasks into a single list
        '''
        self.entry_tasks = []
        self.exit_tasks = []
        for task in self.tasks:
            if task.get_parents() == []:
                self.entry_tasks.append(task)
            if task.get_children == []:
                self.exit_tasks.append(task)

    def total_T(self, tasks=None):
        '''
        function to calculate the total time of our application completion
        :param tasks: list of tasks, if not provided will use self.sorted_priority_tasks
        :return: total time float
        '''
        if tasks is None:
            tasks = self.sorted_priority_tasks
        self.total_t = 0
        for task in tasks:
            if not task.get_children():
                self.total_t = max([self.total_t, task.local_finish_time, task.c_recieve_finish_time])
        return self.total_t

    def total_E(self, tasks=None, power_usage=[1, 2, 4, 0.5]):
        '''
        function which calculates the total energy used by a given set of assigned tasks
        :param tasks: list of task objects
        :param power_usage: energy usage as follows [core1, core2, core3]
        :return: total energy used
        '''
        e_total = 0
        if tasks is None:
            tasks = self.sorted_priority_tasks
        for task in tasks:
            if task.is_core:
                e_task = power_usage[task.core_assigned] * task.core_speed[task.core_assigned]
                e_total += e_task
                #print(task.task_id, e_task)
            elif not task.is_core:
                e_task = power_usage[3] * task.cloud_speed[0]
                e_total += e_task
                #print(task.task_id, e_task)
            else:
                raise ApplicationError("task.is_core is not defined")
        return e_total

    '''
    some getter and setter functions (kinda went out the window at some point)
    '''

    def set_entry_tasks_ready_time(self, time: int):
        for task in self.entry_tasks:
            task.local_ready_time = time

    def set_exe_selection_sequence(self, sequence):
        self.exe_selection_sequence = sequence

    def get_exe_selection_sequence(self):
        return self.exe_selection_sequence

    def get_tasks(self):
        return self.tasks

    def get_entry_tasks(self):
        return self.entry_tasks

    def get_exit_tasks(self):
        return self.exit_tasks

    def set_sorted_priority_tasks(self, sorted_tasks):
        self.sorted_priority_tasks = sorted_tasks

    def get_sorted_priority_tasks(self):
        return self.sorted_priority_tasks


class Task():
    '''
    Node data structure for use with Application, TaskScheduler, and
    TaskMigration objects
    '''
    def __init__(self, task_id: int,
                 parents: list,
                 children: list,
                 cloud_speed: list,
                 local_core_speeds: list):
        # inputs
        self.task_id = task_id
        self.parents = parents
        self.children = children
        self.cloud_speed = cloud_speed
        self.core_speed = local_core_speeds

        # local properties
        self.local_finish_time = 0
        self.c_send_finish_time = 0
        self.c_run_finish_time = 0
        self.c_recieve_finish_time = 0
        self.local_ready_time = -1
        self.c_send_ready_time = -1
        self.c_run_ready_time = -1
        self.c_recieve_ready_time = -1
        self.is_core = None
        self.priority_score = None
        self.core_assigned = None # core1=0, core2=1, core3=2, cloud=3
        self.start_time = [-1, -1, -1, -1] # [core1, core2, core3, cloud]
        self.is_scheduled = None

    def _set_parents(self, tasks: list):
        self.parents = [tasks[i - 1] for i in self.parents]

    def _set_children(self, tasks: list):
        self.children = [tasks[i - 1] for i in self.children]

    def get_parents(self):
        return self.parents

    def get_children(self):
        return self.children

    def set_computation_cost(self, cost):
        self.cost = cost

    def get_computation_cost(self):
        return self.cost

    def get_local_core_speeds(self):
        return self.core_speed

    def get_cloud_speeds(self):
        return self.cloud_speed

    def set_priority_score(self, priority_score: float):
        self.priority_score = priority_score

    def get_priority_score(self):
        return self.priority_score

class TaskScheduler():
    '''
    Initial scheduler for minimizing total execution time.
    Here we generate the minimal-delay schedule without considering
    the energy consumption of the mobile device
    Phase 1: primary assignment
    Phase 2: task prioritization
    Phase 3: execution unit selection
    '''

    def __init__(self, application_object):
        self.app = application_object
        self.primary_assignment()
        self.task_prioritization()
        task_list = self.app.get_sorted_priority_tasks()
        self.execution_unit_selection()
        print(f"initial time {app.total_T(task_list)}; initial energy: {app.total_E(task_list)}")

    def primary_assignment(self):
        '''
        this function determines the subset of tasks that are initially assigned
        for the cloud execution resulting in savings of the application
        completion time.
        :logic: for each task, we find the minimum local execution time (T_l)and compare
        that with the estimated cloud execution time (T_re)
        '''
        for task in self.app.get_tasks():
            t_l_min = task.core_speed[2]
            t_c_min = sum(task.cloud_speed)
            task.is_core = t_l_min < t_c_min


    def task_prioritization(self):
        '''
        initally prioritize tasks in the application for running
        '''

        for task in self.app.get_tasks():
            if not task.is_core:
                task.set_computation_cost(sum(task.get_cloud_speeds()))
            elif task.is_core:
                local_cost = sum(task.get_local_core_speeds()) / len(task.get_local_core_speeds())
                task.set_computation_cost(local_cost)

        for task in self.app.get_tasks():
            task.set_priority_score(None)

        def calculate_priority(task):
            '''
            function to calculate priority scores using recursion, equasions
            :param task:
            :return:
            '''
            if task.get_priority_score() is not None:
                return task.get_priority_score()

            if not task.get_children:
                task.set_priority_score(task.get_computation_cost())
            else:
                child_priorities = [calculate_priority(child) for child in task.get_children()]
                if child_priorities:
                    task.set_priority_score(task.get_computation_cost() + max(child_priorities))
                else:
                    task.set_priority_score(task.get_computation_cost())

            return task.get_priority_score()

        # calculate the priority for tasks starting with exit tasks
        for task in self.app.get_exit_tasks():
            calculate_priority(task)

        # make sure all tasks have a priority score
        for task in self.app.get_tasks():
            if task.get_priority_score() is None:
                calculate_priority(task)

        # sort tasks by their priority scores in descending order
        sorted_tasks = sorted(self.app.get_tasks(), key=lambda x: x.get_priority_score(), reverse=True)
        self.app.set_sorted_priority_tasks(sorted_tasks)


    def process_core_task(self, task):
        '''
        process core task
        :param task: task object
        '''

        if not task.parents:
            task.local_ready_time = 0
        else:
            # equation 3
            for parent in task.parents:
                parent_finish = max(parent.local_finish_time, parent.c_recieve_finish_time)
                if parent_finish > task.local_ready_time:
                    task.local_ready_time = parent_finish

        core_id = 0
        local_done = max(self.local_finish_times[0], task.local_ready_time) + task.core_speed[0]
        if local_done > max(self.local_finish_times[1], task.local_ready_time) + task.core_speed[1]:
            local_done = max(self.local_finish_times[1], task.local_ready_time) + task.core_speed[1]
            core_id = 1
        if local_done > max(self.local_finish_times[2], task.local_ready_time) + task.core_speed[2]:
            local_done = max(self.local_finish_times[2], task.local_ready_time) + task.core_speed[2]
            core_id = 2
        task.core_assigned = core_id
        self.core1.append(task.task_id) if task.core_assigned == 0 else None
        self.core2.append(task.task_id) if task.core_assigned == 1 else None
        self.core3.append(task.task_id) if task.core_assigned == 2 else None

        self.local_finish_times[core_id] = task.local_finish_time = local_done
        task.start_time[core_id] = self.local_finish_times[core_id] - task.core_speed[core_id]


        print(f"task id:{task.task_id}, core:{task.core_assigned + 1}, local start_time: {task.start_time[task.core_assigned]}, finish time: {self.local_finish_times[task.core_assigned]}")

    def process_cloud_task(self, task):
        '''
        process cloud task properties to handle timing and core_assigned
        :param task: task object
        '''

        for parent in task.parents:
            parent_c_send = max(parent.local_finish_time, parent.c_send_finish_time)
            if parent_c_send > task.c_send_ready_time:
                task.c_send_ready_time = parent_c_send
        cloud_ws_finishtime = max(self.cloud_finish_times[0], task.c_send_ready_time) + task.cloud_speed[0]
        task.c_send_finish_time = cloud_ws_finishtime

        # equation 5
        parent_max_finish_time_cloud = 0
        for parent in task.parents:
            if parent.c_run_finish_time > parent_max_finish_time_cloud:
                parent_max_finish_time_cloud = parent.c_run_finish_time
        task.c_run_ready_time = max(task.c_send_finish_time, parent_max_finish_time_cloud)
        cloud_c_finishtime = max(self.cloud_finish_times[1], task.c_run_ready_time) + task.cloud_speed[1]
        task.c_run_finish_time = cloud_c_finishtime

        # equation 6
        task.c_recieve_ready_time = task.c_run_finish_time
        cloud_wr_finishtime = max(self.cloud_finish_times[2], task.c_recieve_ready_time) + task.cloud_speed[2]
        task.c_recieve_finish_time = cloud_wr_finishtime
        task.core_assigned = 3  # 3 is cloud
        task.start_time[3] = max(self.cloud_finish_times[0],task.c_send_ready_time)

        self.cloud_finish_times[0] = cloud_ws_finishtime
        self.cloud_finish_times[1] = cloud_c_finishtime
        self.cloud_finish_times[2] = cloud_wr_finishtime

        self.cloud_seq.append(task.task_id)
        print(f"task id:{task.task_id}; core:{task.core_assigned + 1}, cloud start time: {task.start_time[3]}, finish time: {task.c_recieve_finish_time}")



    def execution_unit_selection(self):
        '''

        :return:
        '''

        self.local_finish_times, self.cloud_finish_times = [0, 0, 0], [0, 0, 0]
        self.core1, self.core2, self.core3, self.cloud_seq = [], [], [], []
        tasks = app.get_sorted_priority_tasks()

        for i, task in enumerate(tasks):
            if task.is_core:
                self.process_core_task(task)
            elif not task.is_core:
                self.process_cloud_task(task)

        seq = [self.core1, self.core2, self.core3, self.cloud_seq]
        #print(f"sequence: {seq}")
        app.set_exe_selection_sequence(seq)



class TaskMigration():
    '''
    task migration for minimizing the energy consumption
    '''

    def __init__(self, application: object):
        self.app = application
        self.outer_loop()

    def update_app(self):
        for task in app.get_tasks():
            pass


    def outer_loop(self):
        '''

        :return:
        '''

        task_list = self.app.get_sorted_priority_tasks()
        sequence = self.app.get_exe_selection_sequence()
        counter = 0
        #for task in task_list:
            #print(task.task_id)
        while counter < 99:
            E_init = app.total_E(task_list)
            matrix = [[(-1, -1) for j in range(4)] for i in range(len(task_list))]
            migrate = [[] for i in range(len(task_list))]

            for i in range(len(task_list)):
                if task_list[i].core_assigned == 3:
                    current_row_id = task_list[i].task_id - 1
                    current_row_value = [1] * 4
                    migrate[current_row_id] = current_row_value
                else:
                    current_row_id = task_list[i].task_id - 1
                    current_row_value = [0] * 4
                    current_row_value[task_list[i].core_assigned] = 1
                    migrate[current_row_id] = current_row_value


            for n in range(len(migrate)):
                for k in range(len(migrate[n])):
                    if migrate[n][k] == 1:
                        continue
                    tasks_copy = deepcopy(task_list)
                    seq_copy = self.generate_sequence(tasks_copy, n + 1, k, deepcopy(sequence))
                    #print(f"seq_copy {seq_copy}")
                    self.kernel_algorithm(tasks_copy, seq_copy)
                    matrix[n][k] = (app.total_T(tasks_copy), app.total_E(tasks_copy))
                    """
                    for task in tasks_copy:
                        print(f"task.task_id {task.task_id}",
                              f"task.local_finish_time {task.local_finish_time}",
                              f"task.c_send_finish_time {task.c_send_finish_time}",
                              f"task.c_run_finish_time {task.c_send_finish_time}",
                              f"task.c_recieve_finish_time {task.c_recieve_finish_time}",
                              f"task.local_ready_time {task.local_ready_time}",
                              f"task.c_send_ready_time {task.c_send_ready_time}",
                              f"task.c_run_ready_time {task.c_run_ready_time}",
                              f"task.c_recieve_ready_time {task.c_recieve_ready_time}",
                              f"task.is_core {task.is_core}",
                              f"task.core_assigned {task.core_assigned}",
                              f"task.start_time {task.start_time}",
                              f"task.is_scheduled {task.is_scheduled}", sep="\n")
                    """

            n_opt, k_opt = -1, -1

            for i in range(len(matrix)):
                for j in range(len(matrix[i])):
                    n = matrix[i][j]
                    if n[0] > self.app.T_max or n == (-1, -1):
                        continue
                    elif (app.total_E(task_list) - n[1]) / abs(n[0] - app.total_T(task_list) + 0.000001) > -1:
                        n_opt = i
                        k_opt = j

            if n_opt == -1 and k_opt == -1:
                break
            n_opt += 1
            k_opt += 1

            sequence = self.generate_sequence(task_list, n_opt, k_opt - 1, sequence)
            self.kernel_algorithm(task_list, sequence)
            counter += 1
            if E_init - app.total_E(task_list) <= 1:
                break
        '''
        for task in tasks_copy:
            print(f"task.task_id {task.task_id}",
                  f"task.local_finish_time {task.local_finish_time}",
                  f"task.c_send_finish_time {task.c_send_finish_time}",
                  f"task.c_run_finish_time {task.c_send_finish_time}",
                  f"task.c_recieve_finish_time {task.c_recieve_finish_time}",
                  f"task.local_ready_time {task.local_ready_time}",
                  f"task.c_send_ready_time {task.c_send_ready_time}",
                  f"task.c_run_ready_time {task.c_run_ready_time}",
                  f"task.c_recieve_ready_time {task.c_recieve_ready_time}",
                  f"task.is_core {task.is_core}",
                  f"task.core_assigned {task.core_assigned}",
                  f"task.start_time {task.start_time}",
                  f"task.is_scheduled {task.is_scheduled}", sep="\n")
        '''


    def update_temp_tasks(self, task, core, local_finish_times):
        '''
        updates local task properties for temp tasks
        :param task: temp task to update
        :param core: core number between 0 and 2
        :return: task.local_finish_time
        '''

        task.start_time = [-1, -1, -1, -1]
        task.start_time[core] = max(local_finish_times[core], task.local_ready_time)
        task.local_finish_time = task.start_time[core] + task.core_speed[core]
        task.c_send_finish_time = -1
        task.c_run_finish_time = -1
        task.c_recieve_finish_time = -1
        return task.local_finish_time

    def update_cloud_tasks(self, task, cloud_finish_times):
        '''
        update cloud task object parameters
        :param task: task object
        :param cloud_finish_times: list of cloud finish times
        :return:cloud_finish_times updated
        '''
        if len(task.parents) == 0:
            task.c_send_ready_time = 0
        else:
            for parent in task.parents:
                parent_max = max(parent.local_finish_time, parent.c_send_finish_time)
                if parent_max > task.c_send_ready_time:
                    task.c_send_ready_time = parent_max
        task.c_send_finish_time = task.cloud_speed[0] + max(task.c_send_ready_time, cloud_finish_times[0])
        task.start_time = [-1, -1, -1, -1]
        task.start_time[3] = max(task.c_send_ready_time, cloud_finish_times[0])
        cloud_finish_times[0] = task.c_send_finish_time

        parent_max_finish_time_cloud = 0
        for parent in task.parents:
            if parent.c_run_finish_time > parent_max_finish_time_cloud:
                parent_max_finish_time_cloud = parent.c_run_finish_time
        task.c_run_ready_time = max(task.c_send_finish_time, parent_max_finish_time_cloud)
        task.c_run_finish_time = task.cloud_speed[1] + max(task.c_run_ready_time, cloud_finish_times[1])
        cloud_finish_times[1] = task.c_run_finish_time
        task.c_recieve_ready_time = task.c_run_finish_time
        task.c_recieve_finish_time = task.cloud_speed[2] + max(task.c_recieve_ready_time, cloud_finish_times[2])
        task.local_finish_time = -1
        cloud_finish_times[2] = task.c_recieve_finish_time

        return cloud_finish_times

    def flagging(self, tasks_list, ready1, ready2, stack, next_task_id):
        for task in tasks_list:
            flag = 0
            for parent in task.parents:
                if not parent.is_scheduled:
                    flag += 1
                ready1[task.task_id - 1] = flag
            if task.task_id == next_task_id:
                ready2[task.task_id - 1] = 0
        for task in tasks_list:
            if (ready1[task.task_id - 1] == 0) and \
                    (ready2[task.task_id - 1] == 0) and \
                    (not task.is_scheduled) and \
                    (task not in stack):
                stack.append(task)

        return tasks_list, ready1, ready2, stack

    def kernel_algorithm(self, tasks_list, seq_new):
        '''
        kernel algorithm
        :param tasks_list: task list
        :param seq_new: current core sequence: [core1, core2, core3, cloud_seq], each one is a list of tasks
        '''

        local_finish_times = [0, 0, 0]
        cloud_finish_times = [0, 0, 0]
        task_index = {}
        temp_id = 0
        for task in tasks_list:
            task_index[task.task_id] = temp_id
            task.local_ready_time = -1
            task.c_send_ready_time = -1
            task.c_run_ready_time = -1
            task.c_recieve_ready_time = -1
            temp_id += 1

        stack = []
        stack.append(tasks_list[0])
        ready1 = [-1] * len(tasks_list)
        ready2 = [-1] * len(tasks_list)
        ready1[tasks_list[0].task_id - 1] = 0
        for each_seq in seq_new:
            if len(each_seq) > 0:
                ready2[each_seq[0] - 1] = 0

        while len(stack):
            temp_task = stack.pop()
            temp_task.is_scheduled = True
            if temp_task.is_core == True:
                if len(temp_task.parents) == 0:
                    temp_task.local_ready_time = 0
                else:
                    for parent in temp_task.parents:
                        p_ft = max(parent.local_finish_time, parent.c_recieve_finish_time)
                        if p_ft > temp_task.local_ready_time:
                            temp_task.local_ready_time = p_ft

            if temp_task.core_assigned in [0, 1, 2]:
                local_finish_times[temp_task.core_assigned] = self.update_temp_tasks(temp_task, temp_task.core_assigned, local_finish_times)

            elif temp_task.core_assigned == 3:
                cloud_finish_times = self.update_cloud_tasks(temp_task, cloud_finish_times)

            seq_temp = seq_new[temp_task.core_assigned]
            temp_task_index = seq_temp.index(temp_task.task_id)
            if temp_task_index != (len(seq_temp) - 1):
                next_task_id = seq_temp[temp_task_index + 1]
            else:
                next_task_id = -1

            tasks_list, ready1, ready2, stack = self.flagging(tasks_list, ready1, ready2, stack, next_task_id)

        for task in tasks_list:
            task.is_scheduled = None

        return tasks_list

    def generate_sequence(self, tasks, target_id, k, sequence):
        '''
        compute new scheduling seq
        :param tasks: list of task objects
        :param target_id: index of targeted task
        :param k: migration location: [0, 1, 2, 3] means: core1, core2, core3, cloud
        :param sequence: current core sequence: [core1, core2, core3, cloud_seq], each one is a list of task_ids
        :return:
        '''
        #print(f"target_id {target_id}, K = {k} sequence = {sequence}")
        task_index, temp_id = {}, 0
        s_new, s_new_prim = sequence[k], []
        for task in tasks:
            task_index[task.task_id] = temp_id
            temp_id += 1
            if task.task_id == target_id:
                target = task

        if target.is_core == True:
            target_rt = target.local_ready_time
        if target.is_core == False:
            target_rt = target.c_send_ready_time

        sequence[target.core_assigned].remove(target.task_id)
        #print(f"sequence: after removing target.taskid = {sequence}, s_new = {s_new}")

        flag = False
        for task_id in s_new:
            task = tasks[task_index[task_id]]
            #print(f"taskid = {task.task_id}, task.start_time {task.start_time}, is_core {task.is_core} ")
            if task.start_time[k] < target_rt:
                s_new_prim.append(task.task_id)
            elif task.start_time[k] >= target_rt and flag == False:
                s_new_prim.append(target.task_id)
                flag = True
            if task.start_time[k] >= target_rt and flag == True:
                s_new_prim.append(task.task_id)
            #print(f"task_id = {task.task_id} ; s_new_prim {s_new_prim}")
        #print(f"sequence: after flagging = {sequence}")
        #print(f"s_new_prim after flagging = {s_new_prim}")
        if not flag:
            s_new_prim.append(target.task_id)
        #print(f"s_new_prim after if not flag: = {s_new_prim}")
        #print(f"s_new_prim {s_new_prim}")
        target.core_assigned = k

        sequence[k] = s_new_prim

        if k == 3:
            target.is_core = False
        else:
            target.is_core = True
        #print(f"final sequence {sequence}")
        return sequence



if __name__ == '__main__':

    task_ids_test1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    local_execution_times_test1 = [[9,8,6,7,5,7,8,6,5,7],
                                   [7,6,5,5,4,6,5,4,3,4],
                                   [5,5,4,3,2,4,3,2,2,2]]
    parents_test1 = [[ ],
                    [1],
                    [1],
                    [1],
                    [1],
                    [1],
                    [3],
                    [2,4,6],
                    [2,4,5],
                    [7,8,9]]

    children_test1 = [[2,3,4,5,6],
                      [8,9],
                      [7],
                      [8,9],
                      [9],
                      [8],
                      [10],
                      [10],
                      [10],
                      []]

    T_maximum_test1 = 27

    task_ids_test2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    local_execution_times_test2 = [[9, 8, 6, 7, 5, 7, 8, 6, 5, 7],
                                   [7, 6, 5, 5, 4, 6, 5, 4, 3, 4],
                                   [5, 5, 4, 3, 2, 4, 3, 2, 2, 2]]
    parents_test2 = [[],
                     [1],
                     [1],
                     [1],
                     [1],
                     [2],
                     [2, 3],
                     [3, 4],
                     [4, 5],
                     [6, 7, 8, 9]]

    children_test2 = [[2, 3, 4, 5],
                      [6, 7],
                      [7, 8],
                      [8, 9],
                      [9],
                      [10],
                      [10],
                      [10],
                      [10],
                      []]

    T_maximum_test2 = 27


    task_ids_test3 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    local_execution_times_test3 = [[9, 8, 6, 7, 5, 7, 8, 6, 5, 7, 9, 8, 6, 7, 5, 7, 8, 6, 5, 7],
                                   [7, 6, 5, 5, 4, 6, 5, 4, 3, 4, 7, 6, 5, 5, 4, 6, 5, 4, 3, 4],
                                   [5, 5, 4, 3, 2, 4, 3, 2, 2, 2, 5, 5, 4, 3, 2, 4, 3, 2, 2, 2]]
    parents_test3 = [[],
                     [1],
                     [1],
                     [1],
                     [2],
                     [2, 3, 4],
                     [3, 4],
                     [4],
                     [5],
                     [5, 6, 7],
                     [7, 8],
                     [8],
                     [9],
                     [10, 11],
                     [15],
                     [13, 14],
                     [15],
                     [16],
                     [16, 17],
                     [18, 19]]

    children_test3 = [[2, 3, 4],
                      [5, 6],
                      [6, 7],
                      [6, 7, 8],
                      [9, 10],
                      [10],
                      [10, 11],
                      [11, 12],
                      [13],
                      [14],
                      [14],
                      [15],
                      [16],
                      [16],
                      [17],
                      [18, 19],
                      [19],
                      [20],
                      [20],
                      []]

    T_maximum_test3 = 53

    task_ids_test4 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    local_execution_times_test4 = [[9, 8, 6, 7, 5, 7, 8, 6, 5, 7, 9, 8, 6, 7, 5, 7, 8, 6, 5, 7],
                                   [7, 6, 5, 5, 4, 6, 5, 4, 3, 4, 7, 6, 5, 5, 4, 6, 5, 4, 3, 4],
                                   [5, 5, 4, 3, 2, 4, 3, 2, 2, 2, 5, 5, 4, 3, 2, 4, 3, 2, 2, 2]]
    parents_test4 = [[],
                     [],
                     [1],
                     [1, 2],
                     [3],
                     [3, 4],
                     [3, 4],
                     [4],
                     [5],
                     [5, 6, 7],
                     [7, 8],
                     [8],
                     [9],
                     [10, 11],
                     [15],
                     [13, 14],
                     [15],
                     [16],
                     [16, 17],
                     [18, 19]]

    children_test4 = [[3, 4],
                      [4],
                      [5, 6, 7],
                      [6, 7, 8],
                      [9, 10],
                      [10],
                      [10, 11],
                      [11, 12],
                      [13],
                      [14],
                      [14],
                      [15],
                      [16],
                      [16],
                      [17],
                      [18, 19],
                      [19],
                      [20],
                      [20],
                      []]

    T_maximum_test4 = 53

    task_ids_test5 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    local_execution_times_test5 = [[9, 8, 6, 7, 5, 7, 8, 6, 5, 7, 9, 8, 6, 7, 5, 7, 8, 6, 5, 7],
                                   [7, 6, 5, 5, 4, 6, 5, 4, 3, 4, 7, 6, 5, 5, 4, 6, 5, 4, 3, 4],
                                   [5, 5, 4, 3, 2, 4, 3, 2, 2, 2, 5, 5, 4, 3, 2, 4, 3, 2, 2, 2]]
    parents_test5 = [[],
                     [],
                     [1],
                     [1, 2],
                     [3],
                     [3, 4],
                     [3, 4],
                     [4],
                     [5],
                     [5, 6, 7],
                     [7, 8],
                     [8],
                     [9],
                     [10, 11],
                     [15],
                     [13, 14],
                     [15],
                     [16],
                     [16, 17],
                     [17]]

    children_test5 = [[3, 4],
                      [4],
                      [5, 6, 7],
                      [6, 7, 8],
                      [9, 10],
                      [10],
                      [10, 11],
                      [11, 12],
                      [13],
                      [14],
                      [14],
                      [15],
                      [16],
                      [16],
                      [17],
                      [18, 19],
                      [19, 20],
                      [],
                      [],
                      []]

    T_maximum_test5 = 53

    cloud_execution_times = [3, 1, 1]
    e_usage = [1, 2, 4, 0.5]
    start = time.perf_counter()

    app = Application(task_ids=task_ids_test1,
                      local_execution_times=local_execution_times_test1,
                      cloud_execution_times=cloud_execution_times,
                      parents=parents_test1,
                      children=children_test1,
                      T_max=T_maximum_test1,
                      energy_usage=e_usage)
    app.set_entry_tasks_ready_time(0)
    TaskScheduler(app)
    TaskMigration(app)

    print("Finish Time:", (time.perf_counter() - start))
    print("final sequence: ", app.get_exe_selection_sequence() )
    print(f"final time: {app.total_T(app.get_sorted_priority_tasks())}, final energy: {app.total_E(app.get_sorted_priority_tasks())}")
    for task in app.get_tasks():
        if task.is_core:
            print(f"task id:{task.task_id}, core:{task.core_assigned + 1}, local start_time: {task.start_time[task.core_assigned]}")
        else:
            print(f"task id:{task.task_id}; core:{task.core_assigned + 1}, cloud start time: {task.start_time[3]}, local start_time: {task.start_time}")
    '''
    for task in app.get_tasks():
        print(f"task.task_id {task.task_id}",
              f"task.local_finish_time {task.local_finish_time}",
              f"task.c_send_finish_time {task.c_send_finish_time}",
              f"task.c_run_finish_time {task.c_send_finish_time}",
              f"task.c_recieve_finish_time {task.c_recieve_finish_time}",
              f"task.local_ready_time {task.local_ready_time}",
              f"task.c_send_ready_time {task.c_send_ready_time}",
              f"task.c_run_ready_time {task.c_run_ready_time}",
              f"task.c_recieve_ready_time {task.c_recieve_ready_time}",
              f"task.is_core {task.is_core}",
              f"task.core_assigned {task.core_assigned}",
              f"task.start_time {task.start_time}",
              f"task.is_scheduled {task.is_scheduled}", sep="\n")
    '''