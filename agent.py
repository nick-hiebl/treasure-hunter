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

WORLD_SIZE = 160

class World:
    def __init__(self):
        self.world = [['' for _ in range(WORLD_SIZE)]
            for _ in range(WORLD_SIZE)]
        self.parent = [[(j, i) for j in range(WORLD_SIZE)]
            for i in range(WORLD_SIZE)]

        self.water = {}
        self.n_stones = {}
        self.n_trees = {}
        self.zones = []

        self.XMIN = WORLD_SIZE
        self.YMIN = WORLD_SIZE
        self.XMAX = 0
        self.YMAX = 0

    def access(self, position):
        return self.world[position[1]][position[0]]

    def set(self, position, item):
        self.world[position[1]][position[0]] = item

    def find_any(self, objective, permitted, start, walkable=True, seen=None):
        if seen == None:
            seen = {}

        if start in seen:
            return None
        seen[start] = True

        if walkable and not permitted(self, start):
            return None

        if objective(self, start):
            return [start]

        if not permitted(self, start):
            return None

        a, b = start
        neighbours = [(a, b - 1), (a + 1, b), (a, b + 1), (a - 1, b)]

        for n in neighbours:
            p = find_any(self, objective, permitted, n, walkable, seen)
            if p:
                return p + [start]

        return None

    def find_nearest(self, objective, permitted, start, walkable=True, limit=100000):
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

            if walkable and not permitted(self, current):
                continue

            if objective(self, current):
                nearest = current
                break

            if not permitted(self, current):
                continue

            a, b = current
            neighbours = [(a, b - 1), (a + 1, b),
                (a, b + 1), (a - 1, b)]

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

    def find_nearest_paths(self, objective, permitted, start, walkable=True, limit=100000):
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

            if walkable and not permitted(self, current):
                continue

            if objective(self, current):
                length = l
                nearest.append(current)

            if not permitted(self, current):
                continue

            a, b = current
            neighbours = [(a, b - 1), (a + 1, b),
                (a, b + 1), (a - 1, b)]

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

    def needs_visiting(self, position):
        a, b = position
        for i in range(-2, 3):
            for j in range(-2, 3):
                if not self.world[b + j][a + i]:
                    return True
        return False

    def how_many_need_visit(self, position):
        a, b = position
        num = 0
        for i in range(-2, 3):
            for j in range(-2, 3):
                if not self.world[b + j][a + i]:
                    num += 1
        return num

    def display(self, position, direction):
        # time.sleep(0.1)
        s = " "
        for c in range(self.XMIN, self.XMAX + 1):
            s += str(c % 10)
        s += "\n"
        for r in range(self.YMIN, self.YMAX + 1):
            s += str(r % 10)
            for c in range(self.XMIN, self.XMAX + 1):
                if (c, r) == position:
                    s += ['^', '>', 'v', '<'][direction]
                else:
                    if self.world[r][c]:
                        s += self.world[r][c]
                    else:
                        s += '#'
            s += '\n'

        print(s)

    def read_in(self, sock, player):
        position = player.position
        direction = player.direction
        i = 0
        j = 0

        self.XMIN = min(self.XMIN, position[0] - 2)
        self.YMIN = min(self.YMIN, position[1] - 2)
        self.XMAX = max(self.XMAX, position[0] + 2)
        self.YMAX = max(self.YMAX, position[1] + 2)

        # print(data)
        while 1:
            data = sock.recv(100)
            if not data:
                exit()
            for ch in data:
                pos = player.view_to_offset(j - 2, i - 2)
                if (i == 2 and j == 2):
                    if not self.world[position[1]][position[0]]:
                        self.world[position[1]][position[0]] = ' '
                    elif self.world[position[1]][position[0]] in 'ako$':
                        self.world[position[1]][position[0]] = ' '
                    pos = player.view_to_offset(j - 1, i - 2)
                    if not self.world[pos[1]][pos[0]]:
                        self.world[pos[1]][pos[0]] = chr(ch)
                    j += 1
                else:
                    self.world[pos[1]][pos[0]] = chr(ch)
                j += 1
                if j > 4:
                    j = 0
                    i = (i + 1) % 5
            if i == 0 and j == 0:
                return


    def get_root(self, A):
        if self.parent[A[1]][A[0]] == A:
            return A
        root = self.get_root(self.parent[A[1]][A[0]])
        self.parent[A[1]][A[0]] = root
        return root

    def merge(self, A, B):
        if self.get_root(A) != self.get_root(B):
            p = self.parent[A[1]][A[0]]
            self.parent[p[1]][p[0]] = B
            self.parent[A[1]][A[0]] = B

    def set_root(self, A, B):
        p = self.get_root(A)
        self.parent[p[1]][p[0]] = B
        self.parent[A[1]][A[0]] = B

    def flood_fill(self, X):
        if not LAND(self, X): return
        x, y = X
        others = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        for o in others:
            if LAND(self, o) and self.get_root(o) != self.get_root(X):
                self.merge(o, X)

    def zone_has(self, zone, item):
        count = 0
        for i in range(self.XMIN, self.XMAX+1):
            for j in range(self.YMIN, self.YMAX+1):
                if self.get_root((i, j)) == zone and \
                        self.world[j][i] == item:
                    count += 1
        return count

    def zone_needs_visiting(self, zone):
        for i in range(self.XMIN, self.XMAX+1):
            for j in range(self.YMIN, self.YMAX+1):
                if self.needs_visiting((i, j)):
                    return True
        return False

    def zone_plain(self, zone):
        interesting = False
        for i in range(self.XMIN, self.XMAX+1):
            if interesting:
                break
            for j in range(self.YMIN, self.YMAX+1):
                if self.get_root((i, j)) == zone:
                    if self.world[j][i] != ' ' or \
                            self.needs_visiting((i, j)):
                        interesting = True
                        break
        return not interesting

    def zone_just_stones(zone, inventory):
        nonstone = False
        for i in range(self.XMIN, self.XMAX+1):
            if nonstone:
                break
            for j in range(self.YMIN, self.YMAX+1):
                if self.get_root((i, j)) == zone and \
                        (not self.world[j][i] in [' ', 'o']):
                    if self.world[j][i] == 'a' and (not 'a' in inventory):
                        nonstone = True
                        break
                    if self.world[j][i] == 'k' and (not 'k' in inventory):
                        nonstone = True
                        break
        return not nonstone

    def analyse(self, position):
        # Generate a union out of contiguous zones of land
        for i in range(self.XMIN, self.XMAX + 1):
            for j in range(self.YMIN, self.YMAX + 1):
                self.flood_fill((i, j))
        # Collect all sets of land
        self.water.clear()
        self.n_stones.clear()
        self.n_trees.clear()
        for i in range(self.XMIN, self.XMAX + 1):
            for j in range(self.YMIN, self.YMAX + 1):
                if LAND(self, (i, j)):
                    self.water[self.get_root((i, j))] = []
                    self.n_stones[self.get_root((i, j))] = 0
                    self.n_trees[self.get_root((i, j))] = 0
        # Add all water blocks adjacent to land
        for i in range(self.XMIN, self.XMAX + 1):
            for j in range(self.YMIN, self.YMAX + 1):
                if STONE(self, (i, j)):
                    self.n_stones[self.get_root((i, j))] += 1
                if TREE(self, (i, j)):
                    self.n_trees[self.get_root((i, j))] += 1
                if WATER(self, (i, j)):
                    neighbours = [(i + 1, j), (i - 1, j),
                        (i, j + 1), (i, j - 1)]
                    for n in neighbours:
                        if LAND(self, n):
                            self.water[self.get_root(n)].append((i, j))

        self.zones = self.water.keys()

    # Returns tuple, first a boolean, second a list which is the path
    # (do this with stones, a path across water)
    def get_water_actions(self, start, inventory, position):
        in_zone = lambda zone: (lambda w, p: w.get_root(p) == zone)
        options_to = {}

        num = self.n_stones[start] + inventory.count('o')

        limit = num + 1
        if 'w' in inventory:
            limit = 100000

        # Determine closest way to get to all other zones
        for w in self.water[start]:
            # Consider other zones
            for other in self.zones:
                if other == start:
                    continue
                p = self.find_nearest(in_zone(other), WATER, w, \
                    False, limit=limit)
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
            return (len(options_to[other][1]) - 1 <= num,
                options_to[other][1])
        # If there are multiple options to go to
        if len(options_to.keys()) > 1:
            stone_path = []
            # See if there are any legal ways to use stepping stones to get to the next location
            for other in options_to.keys():
                if options_to[other][0] - 1 <= num:
                    stone_path.append(options_to[other])
            # Some exist
            if stone_path:
                # Choose a path using this wonky heuristic
                stone_path.sort(key=lambda x: x[0] - self.zone_has(self.get_root(x[1][0]), '$') - self.zone_has(self.get_root(x[1][0]), 'o'))
                return (True, stone_path[0][1])
            # We couldn't find any, but there is some wood
            if 'w' in inventory:
                w = self.choose_boat_start(start, inventory, position)
                return (False, self.find_nearest(GOTO(w), SAFETOWALK, position, False))
        # If there's nowhere we can go, then just head for the water
        if len(options_to.keys()) == 0:
            w = self.choose_boat_start(start, inventory, position)
            path = self.find_nearest(GOTO(w), SAFETOWALK, position, False)
            return (False, path)
        return (False, None)

    # Choose a place to head into the water from
    # Returns a water tile adjacent to the start zone
    def choose_boat_start(self, start, inventory, position):
        # Try choosing a body of water which is meaningful to enter
        # that we can actually get to
        for w in self.water[start]:
            if self.find_nearest(GOTO(w), SAFETOWALK, position, False):
                landing_zone = self.choose_landing_zone(inventory, w)
                if landing_zone:
                    return w
        # Try choosing any body of water
        for w in self.water[start]:
            landing_zone = self.choose_landing_zone(inventory, w)
            if landing_zone:
                return w
        # Choose any block of water we can walk to
        for w in self.water[start]:
            p = self.find_nearest(WATER, SAFETOWALK, position, False)
            if p:
                return p[0]
        # Choose any block of water we have to chop a tree for
        if 'a' in inventory:
            p = self.find_nearest(WATER, SAFETOTREE, position, False)
            if p:
                return p[0]


    # Determine if a zone has enough stones to reach another zone
    def zone_exitable_by_stones(self, start):
        in_zone = lambda zone: (lambda w, p: self.get_root(p) == zone)
        # Consider all water tiles on this island
        for w in self.water[start]:
            # Consider all other zones
            for zone in self.zones:
                # Skip current zone
                if zone == start:
                    continue

                path = self.find_nearest(in_zone(zone), WATER, w, False, limit=self.n_stones[start]+1)
                if path and len(path) <= self.n_stones[start]:
                    return True
        return False

    def choose_landing_zone(self, inventory, position):
        in_zone = lambda zone: (lambda w, p: self.get_root(p) == zone)
        reachable = []

        # A quick first pass over zones to check if obvious candidates exist
        num_interesting = 0
        for zone in self.zones:
            # If we have the treasure and can take it home, then do it
            if self.get_root((WORLD_SIZE//2, WORLD_SIZE//2)) == zone and '$' in inventory:
                return zone
            if not self.zone_plain(zone):
                num_interesting += 1

        # If there's only one zone, then go there
        if num_interesting == 1:
            for zone in self.zones:
                if not self.zone_plain(zone):
                    return zone

        for zone in self.zones:
            # Check if we can even get there
            if self.find_nearest(in_zone(zone), WATER, position, False):

                leavable = False
                # Decide whether a zone can be left once visited
                # If there are trees (we must have an axe by this point)
                if self.n_trees[zone] > 0:
                    leavable = True
                # If there are stones we can use
                elif self.zone_exitable_by_stones(zone):
                    leavable = True

                # If you can't leave this zone then just forget about it
                if not leavable:
                    continue

                value = 0

                # If there is a key
                if (not 'k' in inventory) and self.zone_has(zone, 'k'):
                    value += 3
                # If there is wood then it's probably a nice place
                if self.n_trees[zone]:
                    value += 2
                # If there is the treasure then that's of marginal value
                if self.zone_has(zone, '$'):
                    value += 0.1

                # If zone has stuff I haven't seen then that's great
                if self.zone_needs_visiting(zone):
                    value += 5

                # If there are some rocks so we can probably do stuff
                if self.n_stones[zone]:
                    value += 1

                if value:
                    reachable.append((value, zone))
        if not reachable:
            return None
        reachable.sort(key=lambda x: -x[0])
        return reachable[0][1]

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

    def get_direction_to(self, after):
        x, y = after
        if self.position == (x + 1, y):
            return 3
        if self.position == (x - 1, y):
            return 1
        if self.position == (x, y + 1):
            return 0
        if self.position == (x, y - 1):
            return 2

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
                if not WATER(self.world, self.position):
                    self.on_boat = False
            if self.world.access(self.position) in 'ako$':
                self.inventory.append(self.world.access(self.position))

            if self.world.access(self.position) == '~':
                if self.on_boat:
                    pass
                elif not 'o' in self.inventory:
                    self.on_boat = True
                    self.inventory.remove('w')
                else:
                    self.world.set(self.position, 'O')
                    self.inventory.remove('o')

        elif action in 'lL':
            self.direction = (self.direction + 3) % 4
        elif action in 'rR':
            self.direction = (self.direction + 1) % 4

    def print_world(self):
        self.world.display(self.position, self.direction)
        print("holding [", ", ".join(self.inventory), "]")

GOTO = lambda x: (lambda w, p: p == x)
DISTTO = lambda x: (lambda w, p: abs(p[0] - x[0]) + abs(p[1] - x[1]))

SAFETOWALK = lambda w, p: w.access(p) in [' ', 'O', 'k', 'a', '$']
SAFETODOOR = lambda w, p: w.access(p) in [' ', 'O', 'k', 'a', '-', '$']
DOOR = lambda w, p: w.access(p) == '-'
SAFETOTREE = lambda w, p: w.access(p) in [' ', 'O', 'k', 'a', 'T', '$']
TREE = lambda w, p: w.access(p) == 'T'
SAFETOSTONE = lambda w, p: w.access(p) in [' ', 'O', 'k', 'a', 'o', '$']
STONE = lambda w, p: w.access(p) == 'o'
SAFETOGOLD = lambda w, p: w.access(p) in [' ', 'O', 'k', 'a', 'o', '$']
SAFETOGOLDWITHAXE = lambda w, p: w.access(p) in [' ', 'O', 'k', 'a', 'T', '$']
GOLD = lambda w, p: w.access(p) == '$'

NEEDSVISITING = lambda w, p: w.needs_visiting(p)
VISITVALUE = lambda w, p: w.how_many_need_visit(p)

AXE = lambda w, p: w.access(p) == 'a'
KEY = lambda w, p: w.access(p) == 'k'

LAND = lambda w, p: w.access(p) in [' ', 'o', 'k', 'a', 'O', '$', 'T', '-']
WATER = lambda w, p: w.access(p) == '~'

def run_ai():
    moves = 0

    origin = (WORLD_SIZE//2, WORLD_SIZE//2)

    world = World()
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

    world.read_in(sock, player)
    while 1:
        path = []

        if needs_analysis:
            world.analyse(player.position)
            analysed = True
            needs_analysis = False

        if not player.on_boat:
            # Take the gold home
            if '$' in player.inventory:
                # # TODO: Make it so that you can chop trees on the way home
                if player.has('a'):
                    path = world.find_nearest(GOTO((WORLD_SIZE//2, WORLD_SIZE//2)), SAFETOTREE, player.position)
                else:
                    path = world.find_nearest(GOTO((WORLD_SIZE//2, WORLD_SIZE//2)), SAFETOWALK, player.position)
            # Find the gold
            if not path:
                if player.has('a'):
                    path = world.find_nearest(GOLD, SAFETOGOLDWITHAXE, player.position)
                else:
                    path = world.find_nearest(GOLD, SAFETOGOLD, player.position)
            # Try cutting a tree
            if not path and player.has('a') and not player.has('w'):
                path = world.find_nearest(TREE, SAFETOWALK, player.position, False)
            # Try unlocking a door
            if not path and player.has('k'):
                path = world.find_nearest(DOOR, SAFETOWALK, player.position, False)
            # Try picking up an axe
            if not path and (not player.has('a')):
                path = world.find_nearest(AXE, SAFETOWALK, player.position, True)
            # Try picking up a key
            if not path and (not player.has('k')):
                path = world.find_nearest(KEY, SAFETOWALK, player.position, True)

            # Try visiting somewhere that will show new info
            if not path:
                # Consider all shortest paths to locations that reveal new stuff
                paths = world.find_nearest_paths(NEEDSVISITING, SAFETOWALK, player.position)
                if paths:
                    # Choose the path that reveals the most info
                    val = -VISITVALUE(world, paths[0][0])
                    best = paths[0]
                    for thing in paths:
                        v = -VISITVALUE(world, thing[0])
                        if v < val:
                            best = thing
                    path = best

            # At this point we need to ensure that the world has been analysed for next steps
            if not path:
                world.analyse(player.position)
                analysed = True

            # Try determining what to do next in terms of actions over water
            if not path:
                possibly_stone, a = world.get_water_actions(world.get_root(player.position), player.inventory, player.position)
                # TODO: SOLVE LEVEL s22.in
                print(a)
                if a:
                    if not possibly_stone:
                        path = world.find_nearest(GOTO(a[-1]), SAFETOWALK, player.position, False)
                    elif player.has('o'):
                        # Place a stone
                        path = world.find_nearest(GOTO(a[-1]), SAFETOSTONE, player.position, False)
                    else:
                        # Try doing it with stones
                        if world.n_stones[world.get_root(player.position)] + 1 >= len(a):
                            # Get a stone
                            path = world.find_nearest(STONE, SAFETOSTONE, player.position)
                        else:
                            # Going by boat
                            path = world.find_nearest(GOTO(a[-1]), LAND, player.position, False)

            # If I can't do ANYTHING else, try cutting a tree even though I already have wood
            if not path and player.has('a'):
                path = world.find_nearest(TREE, SAFETOWALK, player.position, False)
        else:
            # Try visiting everywhere in the water that you can
            paths = world.find_nearest_paths(NEEDSVISITING, WATER, player.position)
            if paths:
                # Choose the path that reveals the most info
                val = -VISITVALUE(world, paths[0][0])
                best = paths[0]
                for thing in paths:
                    v = -VISITVALUE(world, thing[0])
                    if v < val:
                        best = thing
                path = best

            if not path:
                target_zone = world.choose_landing_zone(player.inventory, player.position)
                path = world.find_nearest(lambda w, p: world.get_root(p) == target_zone and not TREE(w, p), WATER, player.position, False)

        if not path:
            print("Well this is it then. I have failed.")
            raise Error

        while path:
            first = path.pop()

            actions = []

            way = player.get_direction_to(first)
            print(way)

            actions = player.get_direction_to_moves(way)

            # Change the action if walking into an obstacle
            if TREE(world, first):
                actions.remove('f')
                actions.append('c')
                actions.append('f')
            if DOOR(world, first):
                actions.remove('f')
                actions.append('u')
                actions.append('f')
            if WATER(world, first):
                analysed = False
                needs_analysis = True

            for action in actions:

                moves += 1

                player.update_state(action)
                sock.send(action.encode('utf-8'))
                world.read_in(sock, player)
                player.print_world()
                print(moves)

    sock.close()

if __name__ == "__main__":
    # cProfile.run('run_ai()')
    run_ai()
