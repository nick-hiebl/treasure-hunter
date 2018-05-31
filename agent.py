#!/usr/bin/python3
# ^^ note the python directive on the first line
# COMP 9414 agent initiation file
# requires the host is running before the agent
# designed for python 3.6
# typical initiation would be (file in working directory, port = 31415)
#        python3 agent.py -p 31415
# created by Leo Hoare
# with slight modifications by Alan Blair

import sys
import socket
from collections import deque
from queue import PriorityQueue
import time
import cProfile

WORLD_SIZE = 100

# declaring visible grid to agent
view = [['' for _ in range(5)] for _ in range(5)]

world = [['' for _ in range(WORLD_SIZE)] for _ in range(WORLD_SIZE)]

parent = [[(j, i) for j in range(WORLD_SIZE)] for i in range(WORLD_SIZE)]

water = {}
n_stones = {}
n_trees = {}

class Player:
    def __init__(self, position, direction, inventory, on_boat, world):
        self.position = position
        self.direction = direction
        self.inventory = inventory
        self.world = world
        self.on_boat = on_boat

    def has(self, thing):
        return thing in self.inventory

    def get_direction_to_moves(self, way):
        if way == self.direction:
            return ['f']
        elif (way + 1) % 4 == self.direction:
            return ['l', 'f']
        elif (way + 3) % 4 == self.direction:
            return ['r', 'f']
        else:
            return ['r', 'r', 'f']

    def view_to_offset(self, x, y):
        if self.direction == 0:
            return (self.position[0] + x, self.position[1] + y)
        elif self.direction == 1:
            return (self.position[0] - y, self.position[1] + x)
        elif self.direction == 2:
            return (self.position[0] - x, self.position[1] - y)
        else:
            return (self.position[0] + y, self.position[1] - x)

    def update_state(self, action):
        if action in 'cC' and 'a' in self.inventory:
            if 'w' not in self.inventory:
                self.inventory.append('w')
        elif action in 'fF':
            self.position = self.view_to_offset(0, -1)
            if self.on_boat:
                if not WATER(self.position):
                    self.on_boat = False
            if self.world[self.position[1]][self.position[0]] in 'ako$':
                self.inventory.append(self.world[self.position[1]][self.position[0]])

                # INVESTIGATE
                self.world[self.position[1]][self.position[0]]
            if self.world[self.position[1]][self.position[0]] == '~':
                if self.on_boat:
                    pass
                elif not 'o' in self.inventory:
                    self.on_boat = True
                    self.inventory.remove('w')
                else:
                    self.world[self.position[1]][self.position[0]] = 'O'
                    self.inventory.remove('o')

        elif action in 'lL':
            self.direction = (self.direction + 3) % 4
        elif action in 'rR':
            self.direction = (self.direction + 1) % 4

    def print_world(self):
        print_world(self.position, self.direction)
        print("holding [", ", ".join(self.inventory), "]")

GOTO = lambda x: (lambda p: p == x)
DISTTO = lambda x: (lambda p: abs(p[0] - x[0]) + abs(p[1] - x[1]))

SAFETOWALK = lambda p: world[p[1]][p[0]] in [' ', 'O', 'k', 'a', '$']
SAFETODOOR = lambda p: world[p[1]][p[0]] in [' ', 'O', 'k', 'a', '-', '$']
DOOR = lambda p: world[p[1]][p[0]] == '-'
SAFETOTREE = lambda p: world[p[1]][p[0]] in [' ', 'O', 'k', 'a', 'T', '$']
TREE = lambda p: world[p[1]][p[0]] == 'T'
SAFETOSTONE = lambda p: world[p[1]][p[0]] in [' ', 'O', 'k', 'a', 'o', '$']
STONE = lambda p: world[p[1]][p[0]] == 'o'
SAFETOGOLD = lambda p: world[p[1]][p[0]] in [' ', 'O', 'k', 'a', 'o', '$']
GOLD = lambda p: world[p[1]][p[0]] == '$'

AXE = lambda p: world[p[1]][p[0]] == 'a'
KEY = lambda p: world[p[1]][p[0]] == 'k'

LAND = lambda p: world[p[1]][p[0]] in [' ', 'o', 'k', 'a', 'O', '$', 'T', '-']
WATER = lambda p: world[p[1]][p[0]] == '~'

XMIN = WORLD_SIZE
YMIN = WORLD_SIZE

XMAX = 0
YMAX = 0

def needs_visiting(position):
    a, b = position
    for i in range(-2, 3):
        for j in range(-2, 3):
            if not world[b + j][a + i]:
                return True
    return False

def how_many_need_visit(position):
    a, b = position
    num = 0
    for i in range(-2, 3):
        for j in range(-2, 3):
            if not world[b + j][a + i]:
                num += 1
    return num

def find_any(objective, permitted, start, walkable=True, seen=None):
    if seen == None:
        seen = {}

    if start in seen:
        return None
    seen[start] = True

    if walkable and not permitted(start):
        return None

    if objective(start):
        return [start]

    if not permitted(start):
        return None

    a, b = start
    neighbours = [(a, b - 1), (a + 1, b), (a, b + 1), (a - 1, b)]

    for n in neighbours:
        p = find_any(objective, permitted, n, walkable, seen)
        if p:
            return p + [start]

    return None

def find_nearest(objective, permitted, start, walkable=True, limit=100000):
    empty = None

    queue = deque()
    queue.append((0, start))

    seen = {}
    prev = {}
    prev[start] = (-1, -1)

    nearest = (-1, -1)

    while len(queue):

        l, current = queue.popleft()

        if l > limit:
            return []

        if current in seen:
            continue
        seen[current] = True

        if walkable and not permitted(current):
            continue

        if objective(current):
            nearest = current
            break

        if not permitted(current):
            continue

        a, b = current
        neighbours = [(a, b - 1), (a + 1, b), (a, b + 1), (a - 1, b)]

        for n in neighbours:
            if (not n in seen) and (not n in prev):
                prev[n] = current
                queue.append((l+1, n))

    if nearest == (-1, -1):
        return []
    else:
        out = [nearest]
        nearest = prev[nearest]
        while nearest != start and nearest != (-1, -1):
            out.append(nearest)
            nearest = prev[nearest]
        return out

def find_nearest_paths(objective, permitted, start, walkable=True, limit=100000):
    queue = deque()
    queue.append((0, start))
    seen = {}
    prev = {}
    prev[start] = (-1, -1)

    nearest = []
    length = -1

    while len(queue):
        l, current = queue.popleft()

        if l > limit:
            return []

        if nearest:
            if l > length:
                break

        if current in seen:
            continue
        seen[current] = True

        if walkable and not permitted(current):
            continue

        if objective(current):
            length = l
            nearest.append(current)

        if not permitted(current):
            continue

        a, b = current
        neighbours = [(a, b - 1), (a + 1, b), (a, b + 1), (a - 1, b)]

        for n in neighbours:
            if (not n in seen) and (not n in prev):
                prev[n] = current
                queue.append((l+1, n))

    if nearest == []:
        return []
    else:
        paths = []
        for end in nearest:
            out = [end]
            end = prev[end]
            while end != start and end != (-1, -1):
                out.append(end)
                end = prev[end]
            paths.append(out)
        return paths

# function to take get action from AI or user
def get_action(view):

    ## REPLACE THIS WITH AI CODE TO CHOOSE ACTION ##

    # input loop to take input from user (only returns if this is valid)
    while 1:
        inp = input("Enter Action(s): ")
        inp.strip()
        final_string = ''
        for char in inp:
            if char in ['f','l','r','c','u','b','F','L','R','C','U','B', '0', '1', '2', '3']:
                final_string += char
        if final_string:
            return list(final_string)

def print_world(position, direction):
    global XMIN, XMAX, YMIN, YMAX
    # time.sleep(0.1)
    print(" ", end="")
    for c in range(XMIN, XMAX + 1):
        print(c % 10, end="")
    print()
    for r in range(YMIN, YMAX + 1):
        print(r % 10, end="")
        for c in range(XMIN, XMAX + 1):
            if (c, r) == position:
                print(['^', '>', 'v', '<'][direction], end="")
            else:
                print(world[r][c] if world[r][c] else '#', end="")
        print()

# helper function to print the grid
def print_grid(view):
    print('\n' * 40)
    print('+-----+')
    for ln in view:
        print("|"+str(ln[0])+str(ln[1])+str(ln[2])+str(ln[3])+str(ln[4])+"|")
    print('+-----+')

def read_in_world(sock, player):
    global XMIN, XMAX, YMIN, YMAX
    position = player.position
    direction = player.direction
    i = 0
    j = 0

    XMIN = min(XMIN, position[0] - 2)
    YMIN = min(YMIN, position[1] - 2)
    XMAX = max(XMAX, position[0] + 2)
    YMAX = max(YMAX, position[1] + 2)

    # print(data)
    while 1:
        data = sock.recv(100)
        if not data:
            exit()
        for ch in data:
            pos = player.view_to_offset(j - 2, i - 2)
            if (i == 2 and j == 2):
                view[i][j] = '^'
                if not world[position[1]][position[0]]:
                    world[position[1]][position[0]] = ' '
                elif world[position[1]][position[0]] in 'ako$':
                    world[position[1]][position[0]] = ' '
                view[i][j + 1] = chr(ch)
                pos = player.view_to_offset(j - 1, i - 2)
                if not world[pos[1]][pos[0]]:
                    world[pos[1]][pos[0]] = chr(ch)
                j += 1
            else:
                view[i][j] = chr(ch)
                world[pos[1]][pos[0]] = chr(ch)
            j += 1
            if j > 4:
                j = 0
                i = (i + 1) % 5
        if i == 0 and j == 0:
            return

def get_direction_to(current, after):
    x, y = current
    if (x - 1, y) == after:
        return 3
    if (x + 1, y) == after:
        return 1
    if (x, y - 1) == after:
        return 0
    if (x, y + 1) == after:
        return 2

def choose_move(move, sock):
    update_state(move)

    sock.send(move.encode('utf-8'))
    read_in_world(sock)

def get_root(A):
    if parent[A[1]][A[0]] == A:
        return A
    root = get_root(parent[A[1]][A[0]])
    parent[A[1]][A[0]] = root
    return root

def merge(A, B):
    if get_root(A) != get_root(B):
        p = parent[A[1]][A[0]]
        parent[p[1]][p[0]] = B
        parent[A[1]][A[0]] = B

def set_root(A, B):
    p = get_root(A)
    parent[p[1]][p[0]] = B
    parent[A[1]][A[0]] = B

def flood_fill(X):
    if not LAND(X): return
    x, y = X
    others = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    for o in others:
        if LAND(o) and get_root(o) != get_root(X): merge(o, X)

def zone_has(zone, item):
    count = 0
    for i in range(XMIN, XMAX+1):
        for j in range(YMIN, YMAX+1):
            if get_root((i, j)) == zone and world[j][i] == item:
                count += 1
    return count

def zone_plain(zone):
    interesting = False
    for i in range(XMIN, XMAX+1):
        if interesting:
            break
        for j in range(YMIN, YMAX+1):
            if get_root((i, j)) == zone and world[j][i] != ' ':
                interesting = True
                break
    return not interesting

def zone_just_stones(zone, inventory):
    nonstone = False
    for i in range(XMIN, XMAX+1):
        if nonstone:
            break
        for j in range(YMIN, YMAX+1):
            if get_root((i, j)) == zone and (not world[j][i] in [' ', 'o']):
                if world[j][i] == 'a' and (not 'a' in inventory):
                    nonstone = True
                    break
                if world[j][i] == 'k' and (not 'k' in inventory):
                    nonstone = True
                    break
    return not nonstone

def choose_boat_start(zones, start, water, n_stones, n_trees, inventory, position):
    for w in water[start]:
        if find_nearest(GOTO(w), SAFETOWALK, position, False):
            landing_zone = choose_landing_zone(zones, water, n_stones, n_trees, inventory, w)
            if landing_zone:
                return w
    for w in water[start]:
        landing_zone = choose_landing_zone(zones, water, n_stones, n_trees, inventory, w)
        if landing_zone:
            return w

def adjacent(a, b):
    x, y = a
    if (x + 1, y) == b or (x - 1, y) == b or (x, y + 1) == b or (x, y - 1) == b:
        return True
    return False

# Returns tuple, first a boolean, second a list which is the path
# (do this with stones, a path across water)
def get_water_actions(zones, start, water, n_stones, n_trees, inventory, position):
    in_zone = lambda zone: (lambda p: get_root(p) == zone)
    options_to = {}

    limit = n_stones[start] + inventory.count('o') + 1
    if 'w' in inventory:
        limit = 100000

    print("Limited to", limit)

    # Determine closest way to get to all other zones
    for w in water[start]:
        # Consider other zones
        for other in zones:
            if other == start:
                continue
            p = []
            p = find_nearest(in_zone(other), WATER, w, False, limit=limit)
            # If we can reach this zone
            if p:
                p.append(w)
                if not other in options_to:
                    options_to[other] = (len(p), p)
                else:
                    if len(p) < options_to[other][0]:
                        options_to[other] = (len(p), p)
    # If there's only one place we can go, then go there
    if len(options_to.keys()) == 1:
        other = list(options_to.keys())[0]
        return (len(options_to[other][1]) - 1 <= n_stones[start] + inventory.count('o'), options_to[other][1])
    # If there are multiple options to go to
    if len(options_to.keys()) > 1:
        stone_path = []
        num = n_stones[start] + inventory.count('o')
        # See if there are any legal ways to use stepping stones to get to the next location
        for other in options_to.keys():
            if options_to[other][0] - 1 <= num:
                stone_path.append(options_to[other])
        # Some exist
        if stone_path:
            # Choose a path using this wonky heuristic
            stone_path.sort(key=lambda x: x[0] - zone_has(get_root(x[1][0]), '$') - zone_has(get_root(x[1][0]), 'o'))
            return (True, stone_path[0][1])
        # We couldn't find any, but there is some wood
        if 'w' in inventory:
            w = choose_boat_start(zones, start, water, n_stones, n_trees, inventory, position)
            return (False, find_nearest(GOTO(w), SAFETOWALK, position, False))
    # If there's nowhere we can go, then just head for the water
    if len(options_to.keys()) == 0:
        print("no options")
        w = choose_boat_start(zones, start, water, n_stones, n_trees, inventory, position)
        print(w)
        path = find_nearest(GOTO(w), SAFETOWALK, position, False)
        print(path)
        return (False, path)
    return (False, None)

# Determine if a zone has enough stones to reach another zone
def zone_exitable_by_stones(zones, start, water, n_stones):
    in_zone = lambda zone: (lambda p: get_root(p) == zone)
    # Consider all water tiles on this island
    for w in water[start]:
        # Consider all other zones
        for zone in zones:
            # Skip current zone
            if zone == start:
                continue

            path = find_nearest(in_zone(zone), WATER, w, False, limit=n_stones[start]+1)
            if path and len(path) <= n_stones[start]:
                return True
    return False

def choose_landing_zone(zones, water, n_stones, n_trees, inventory, position):
    in_zone = lambda zone: (lambda p: get_root(p) == zone)
    reachable = []

    # A quick first pass over zones to check if obvious candidates exist
    num_interesting = 0
    for zone in zones:
        # If we have the treasure and can take it home, then do it
        if get_root((WORLD_SIZE//2, WORLD_SIZE//2)) == zone and '$' in inventory:
            return zone
        if not zone_plain(zone):
            num_interesting += 1

    # If there's only one zone, then go there
    if num_interesting == 1:
        for zone in zones:
            if not zone_plain(zone):
                return zone

    for zone in zones:
        # Check if we can even get there
        if find_nearest(in_zone(zone), WATER, position, False):

            leavable = False
            # Decide whether a zone can be left once visited

            # If we can make another boat, then we're good
            # Apparently we cannot hold multiple rafts, so nevermind
            # if 'w' in inventory:
            #     leavable = True
            # If there are trees (we must have an axe by this point)
            if n_trees[zone] > 0:
                leavable = True
            # If there are stones we can use
            elif zone_exitable_by_stones(zones, zone, water, n_stones):
                leavable = True

            # If you can't leave this zone then just forget about it
            if not leavable:
                continue

            value = 0

            # If there is a key
            if (not 'k' in inventory) and zone_has(zone, 'k'):
                value += 3
            # If there is wood
            if n_trees[zone]:
                value += 2
            # If there is the treasure
            if zone_has(zone, '$'):
                value += 0.1

            # If there are some rocks so we can probably merge things
            if n_stones[zone]:
                value += 1

            if value:
                reachable.append((value, zone))
    if not reachable:
        return None
    reachable.sort(key=lambda x: -x[0])
    return reachable[0][1]


def find_route(zones, start, water, n_stones, n_trees, connected_by):
    in_zone = lambda zone: (lambda p: get_root(p) == zone)

    def state(cost_to, pos, n_stones, n_trees):
        return ()

    def heuristic():
        return 0

    states = PriorityQueue()

def analyse_world(water, n_stones, n_trees, position):
    global XMIN, XMAX, YMIN, YMAX
    # Generate a union out of contiguous zones of land
    for i in range(XMIN, XMAX + 1):
        for j in range(YMIN, YMAX + 1):
            flood_fill((i, j))
    # Collect all sets of land
    water.clear()
    n_stones.clear()
    n_trees.clear()
    for i in range(XMIN, XMAX + 1):
        for j in range(YMIN, YMAX + 1):
            if LAND((i, j)):
                water[get_root((i, j))] = []
                n_stones[get_root((i, j))] = 0
                n_trees[get_root((i, j))] = 0
    # Add all water blocks adjacent to land
    for i in range(XMIN, XMAX + 1):
        for j in range(YMIN, YMAX + 1):
            if STONE((i, j)):
                n_stones[get_root((i, j))] += 1
            if TREE((i, j)):
                n_trees[get_root((i, j))] += 1
            if WATER((i, j)):
                neighbours = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
                for n in neighbours:
                    if LAND(n):
                        water[get_root(n)].append((i, j))

    zones = water.keys()

def run_ai():
    moves = 0

    origin = (WORLD_SIZE//2, WORLD_SIZE//2)

    player = Player(origin, 0, [], False, world)

    analysed = False
    needs_analysis = False

    # checks for correct amount of arguments
    if len(sys.argv) != 3:
        print("Usage Python3 " + sys.argv[0] + " -p port \n")
        sys.exit(1)

    port = int(sys.argv[2])

    # checking for valid port number
    if not 1025 <= port <= 65535:
        print('Incorrect port number')
        sys.exit()

    # creates TCP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
         # tries to connect to host
         # requires host is running before agent
         sock.connect(('localhost',port))
    except (ConnectionRefusedError):
         print('Connection refused, check host is running')
         sys.exit()

    # navigates through grid with input stream of data
    i = 0
    j = 0

    read_in_world(sock, player)
    # print_grid(view) # COMMENT THIS OUT ON SUBMISSION
    while 1:
        path = []

        if needs_analysis:
            analyse_world(water, n_stones, n_trees, player.position)
            analysed = True
            needs_analysis = False

        if not player.on_boat:
            # Take the gold home
            if '$' in player.inventory:
                path = find_nearest(GOTO((WORLD_SIZE//2, WORLD_SIZE//2)), SAFETOWALK, player.position)
            # Find the gold
            if not path:
                path = find_nearest(GOLD, SAFETOGOLD, player.position)
            # Try cutting a tree
            if not path and player.has('a'):
                path = find_nearest(TREE, SAFETOWALK, player.position, False)
            # Try unlocking a door
            if not path and player.has('k'):
                path = find_nearest(DOOR, SAFETOWALK, player.position, False)
            # Try picking up an axe
            if not path and (not player.has('a')):
                path = find_nearest(AXE, SAFETOWALK, player.position, True)
            # Try picking up a key
            if not path and (not player.has('k')):
                path = find_nearest(KEY, SAFETOWALK, player.position, True)

            # Try visiting somewhere that will show new info
            if not path:
                # path = find_nearest(needs_visiting, SAFETOWALK, position)
                paths = find_nearest_paths(needs_visiting, SAFETOWALK, player.position)
                if paths:
                    paths.sort(key=lambda x: -how_many_need_visit(x[0]))
                    path = paths[0]

            # At this point we need to ensure that the world has been analysed for next steps
            if not path:
                analyse_world(water, n_stones, n_trees, player.position)
                analysed = True

            # Try determining what to do next in terms of actions over water
            if not path:
                possibly_stone, a = get_water_actions(list(water.keys()), get_root(player.position), water, n_stones, n_trees, player.inventory, player.position)
                print(a)
                if a:
                    if not possibly_stone:
                        path = find_nearest(GOTO(a[-1]), SAFETOWALK, player.position, False)
                    elif player.has('o'):
                        # Place a stone
                        path = find_nearest(GOTO(a[-1]), SAFETOSTONE, player.position, False)
                    else:
                        # Try doing it with stones
                        if n_stones[get_root(player.position)] + 1 >= len(a):
                            # Get a stone
                            path = find_nearest(STONE, SAFETOSTONE, player.position)
                        else:
                            # Going by boat
                            path = find_nearest(GOTO(a[-1]), LAND, player.position, False)
        else: # in_boat
            paths = find_nearest_paths(needs_visiting, WATER, player.position)
            if paths:
                paths.sort(key=lambda x: -how_many_need_visit(x[0]))
                path = paths[0]

            if not path:
                target_zone = choose_landing_zone(list(water.keys()), water, n_stones, n_trees, player.inventory, player.position)
                path = find_nearest(lambda p: get_root(p) == target_zone, WATER, player.position, False)

        if not path:
            print("Well this is it then. I have failed.")
            raise Error

        while path:
            # first = path[-1]
            first = path.pop()

            actions = []

            way = get_direction_to(player.position, first)

            actions = player.get_direction_to_moves(way)

            if TREE(first):
                actions.remove('f')
                actions.append('c')
            if DOOR(first):
                actions.remove('f')
                actions.append('u')
            if WATER(first):
                analysed = False
                needs_analysis = True

            for action in actions:

                moves += 1

                player.update_state(action)
                sock.send(action.encode('utf-8'))
                read_in_world(sock, player)
                # print_grid(view) # COMMENT THIS OUT ON SUBMISSION
                player.print_world()
                print(moves)

    sock.close()

if __name__ == "__main__":
    # cProfile.run('run_ai()')
    run_ai()
