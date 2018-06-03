#!/usr/bin/python3
# Developed by Nicholas Hiebl z5159918
# Based on template be Leo Hoare, modified by Alan Blair
# Typical usage would be
#       python3 agent.py -p 31415
# or
#       ./agent.py -p 31415
#
# The question:
# Briefly describe how your program works, including any algorithms and
# data structures employed, and explain any design decisions you made
# along the way.
#
#
#
#

import sys
import socket
from collections import deque
from queue import PriorityQueue
import time
import cProfile

WORLD_SIZE = 160

class World:
    def __init__(self):
        # Stores the item at each coordinate
        self.world = [['' for _ in range(WORLD_SIZE)]
            for _ in range(WORLD_SIZE)]

        # Stores the parent of each coordinate
        # This is used for the union find structure which segregates
        # the world into distinct zones of land
        self.parent = [[(j, i) for j in range(WORLD_SIZE)]
            for i in range(WORLD_SIZE)]

        # Stores the water tiles at the edge of each zone
        self.water = {}

        # Stores the number of stones in each zone
        self.n_stones = {}

        # Stores the number of trees in each zone
        self.n_trees = {}

        # Stores the list of zones in the world
        self.zones = []

        # Stores the extreme x and y values used to represent the world
        self.XMIN = WORLD_SIZE
        self.YMIN = WORLD_SIZE
        self.XMAX = 0
        self.YMAX = 0

    # Get the tile at a given pair of coordinates
    def access(self, position):
        return self.world[position[1]][position[0]]

    # Store a value in the world
    def set(self, position, item):
        self.world[position[1]][position[0]] = item

    # Performs a BFS on the map, searching for a place satisfying the
    # objective function whilst only traversing tiles satisfying the
    # permitted function. Starts at the start location* and if walkable
    # is true then the objective must satisfy the objective function.
    #
    # *start can be a list of starting locations to search from
    def find_nearest(self, objective, permitted, start, walkable=True, limit=100000):
        # Use a queue to perform BFS
        queue = deque()

        # Keep track of seen items and previous spaces with a dictionary
        seen = {}
        prev = {}

        # If given a list of start points load them
        if type(start) == list:
            for pos in start:
                prev[pos] = (-1, -1)
                queue.append((0, pos))
        # If given a single start point load that
        else:
            prev[start] = (-1, -1)
            queue.append((0, start))

        # Assume that no target location can be found
        nearest = (-1, -1)

        while len(queue):

            l, current = queue.popleft()

            if l > limit:
                return []

            if current in seen:
                continue
            seen[current] = True

            # Walkable indicates that the target location must
            # satisfy the permitted function as well as the objective
            # function
            if walkable and not permitted(self, current):
                continue

            # Check if a target location has been found
            if objective(self, current):
                nearest = current
                break

            # If the location doesn't satisfy permitted, don't expand
            # its neighbours
            if not permitted(self, current):
                continue

            a, b = current
            # Consider all adjacent locations
            neighbours = [(a, b - 1), (a + 1, b),
                (a, b + 1), (a - 1, b)]

            # Expand all four neighbours into our queue
            for n in neighbours:
                if (not n in seen) and (not n in prev):
                    prev[n] = current
                    queue.append((l + 1, n))

        # If nothing could be found, return an empty path
        if nearest == (-1, -1):
            return []
        else:
            out = [nearest]
            nearest = prev[nearest]
            if type(start) == list:
                # Here we want to ensure we return the start location
                # so that the user knows about it
                while nearest != (-1, -1):
                    out.append(nearest)
                    nearest = prev[nearest]
            else:
                # Do not return the starting location, as since only
                # one location was given, the start must be known
                while nearest != start and nearest != (-1, -1):
                    out.append(nearest)
                    nearest = prev[nearest]
            return out

    # Perform a BFS as with the above function, but return a list of all
    # paths of minimum length
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
                # If we have already found SOME paths and we are now
                # examining a path that is LONGER than that, then we
                # should stop, as we are only trying to find shortest
                # length paths
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

    # Returns true if visiting a position would reveal new locations
    # in the world
    def needs_visiting(self, position):
        a, b = position
        for i in range(-2, 3):
            for j in range(-2, 3):
                if not self.world[b + j][a + i]:
                    return True
        return False

    # Returns the number of locations within a 5x5 of a given position
    # that have not yet been seen
    def how_many_need_visit(self, position):
        a, b = position
        num = 0
        for i in range(-2, 3):
            for j in range(-2, 3):
                if not self.world[b + j][a + i]:
                    num += 1
        return num

    # Print out info about the world
    def display(self, position, direction):
        # Loops use YMIN, YMAX, XMIN, XMAX to ensure only the minimum
        # amount needed to see the whole world is displayed
        s = " "
        # "Axes" displaying the first digit of x, y coordinate are also
        # displayed for debugging help
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

    # Load up info about the world from a socket and store it in our
    # world grid to update the world model
    def read_in(self, sock, agent):
        position = agent.position
        direction = agent.direction
        i = 0
        j = 0

        self.XMIN = min(self.XMIN, position[0] - 2)
        self.YMIN = min(self.YMIN, position[1] - 2)
        self.XMAX = max(self.XMAX, position[0] + 2)
        self.YMAX = max(self.YMAX, position[1] + 2)

        # Code copied basically from template
        while 1:
            data = sock.recv(100)
            if not data:
                exit()
            for ch in data:
                # Transfer coordinates from being relative to view
                # to being absolute coordinates in the world
                pos = agent.view_to_offset(j - 2, i - 2)
                if (i == 2 and j == 2):
                    if not self.world[position[1]][position[0]]:
                        self.world[position[1]][position[0]] = ' '
                    elif self.world[position[1]][position[0]] in 'ako$':
                        self.world[position[1]][position[0]] = ' '
                    pos = agent.view_to_offset(j - 1, i - 2)
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

    # Using union find data structure to determine what "zone" a given
    # coordinate is in
    def get_root(self, A):
        if self.parent[A[1]][A[0]] == A:
            return A
        root = self.get_root(self.parent[A[1]][A[0]])
        self.parent[A[1]][A[0]] = root
        return root

    # Use union find data structure to combine two disparate world
    # regions into one
    def merge(self, A, B):
        if self.get_root(A) != self.get_root(B):
            p = self.parent[A[1]][A[0]]
            self.parent[p[1]][p[0]] = B
            self.parent[A[1]][A[0]] = B

    # Union find data structure function to set the "root" or "zone"
    # that a given location belongs to
    def set_root(self, A, B):
        p = self.get_root(A)
        self.parent[p[1]][p[0]] = B
        self.parent[A[1]][A[0]] = B

    # Connect a tile to its adjacent neighbours if they're all on land
    def flood_fill(self, X):
        x, y = X
        others = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        if LAND(self, X):
            for o in others:
                if LAND(self, o) and self.get_root(o) != self.get_root(X):
                    self.merge(o, X)

    # Scan through a zone and see if it contains a particular item
    def zone_has(self, zone, item):
        count = 0
        for i in range(self.XMIN, self.XMAX+1):
            for j in range(self.YMIN, self.YMAX+1):
                if self.get_root((i, j)) == zone and \
                        self.world[j][i] == item:
                    count += 1
        return count

    # Check if a zone has any tiles that need revealing without having
    # to clobber anything
    def zone_needs_visiting(self, zone):
        for i in range(self.XMIN, self.XMAX+1):
            for j in range(self.YMIN, self.YMAX+1):
                if self.get_root((i, j)) == zone:
                    if not SAFETOSTONE(self, (i, j)):
                        continue
                    if self.needs_visiting((i, j)):
                        return True
        return False

    # Check if a zone contains no meaningful info or tools
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

    # Check if a zone has nothing other than stones (or duplicate tools)
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

    # Analyse the world state using a union find data structure and
    # create a few important points of data.
    #
    # World.parent stores the union find data structure
    #
    # World.water stores the water tiles which are on the boundary of
    # a given zone
    #
    # World.n_stones and World.n_trees count the number of stones and
    # trees in each zone
    #
    # World.zones stores the list of zones in the world
    def analyse(self, position):
        # Generate a union out of contiguous zones of land
        for i in range(self.XMIN, self.XMAX + 1):
            for j in range(self.YMIN, self.YMAX + 1):
                self.parent[i][j] = (j, i)
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
                    zone = self.get_root((i, j))
                    self.water[zone] = []
                    self.n_stones[zone] = 0
                    self.n_trees[zone] = 0
        # Add all water blocks adjacent to land
        for i in range(self.XMIN, self.XMAX + 1):
            for j in range(self.YMIN, self.YMAX + 1):
                zone = self.get_root((i, j))
                if STONE(self, (i, j)):
                    self.n_stones[zone] += 1
                if TREE(self, (i, j)):
                    self.n_trees[zone] += 1
                if WATER(self, (i, j)):
                    neighbours = [(i + 1, j), (i - 1, j),
                        (i, j + 1), (i, j - 1)]
                    for n in neighbours:
                        if LAND(self, n):
                            if not (i, j) in self.water[self.get_root(n)]:
                                self.water[self.get_root(n)].append((i, j))

        self.zones = self.water.keys()

    # Returns tuple, first a boolean, second a list which is the path
    # (do this with stones, a path across water)
    #
    # If the boolean is False then the trip should be made using a boat
    def get_water_actions(self, start, inventory, position):
        in_zone = lambda zone: (lambda w, p: w.get_root(p) == zone)
        options_to = {}

        num = self.n_stones[start] + inventory.count('o')

        limit = num + 1
        if 'w' in inventory:
            limit = 100000

        # Determine closest way to get to all other zones
        # Consider other zones
        for other in self.zones:
            if other == start:
                continue
            p = self.find_nearest(in_zone(other), WATER, self.water[start], \
                False, limit=limit)
            # If we can reach this zone
            if p:
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
                    # If we are in the home zone and can immediately get to the gold with stones
                    # then go there and come back
                    if self.zone_has(other, '$') and self.get_root((WORLD_SIZE//2, WORLD_SIZE//2)) == self.get_root(position):
                        return (True, options_to[other][1])
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
        options = []
        for w in self.water[start]:
            if self.find_nearest(GOTO(w), SAFETOWALK, position, False):
                landing_zone = self.choose_landing_zone(inventory, w)
                if landing_zone and landing_zone != start:
                    found = False
                    for thing in options:
                        if thing[1] == landing_zone:
                            found = True
                            break
                    if not found:
                        options.append((w, landing_zone))
        if options:
            options.sort(key=lambda x: -self.evaluate_zone(x[1], inventory))
            return options[0][0]

        # Try choosing any body of water
        for w in self.water[start]:
            landing_zone = self.choose_landing_zone(inventory, w)
            if landing_zone and landing_zone != start:
                return w
        # Choose any block of water we can walk to
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

    # Assign a zone a numerical value based on the stuff it has in it
    def evaluate_zone(self, zone, inventory):
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
            value += self.n_stones[zone]

        return value

    # Choose where to go back to land once in the water
    def choose_landing_zone(self, inventory, position):
        in_zone = lambda zone: (lambda w, p: self.get_root(p) == zone)
        reachable = []
        # A quick first pass over zones to check if obvious candidates exist
        num_interesting = 0

        home_zone = self.get_root((WORLD_SIZE//2, WORLD_SIZE//2))

        # If we have the treasure and can take it home, then do it
        if self.find_nearest(in_zone(home_zone), WATER, position, False) \
                and '$' in inventory:
            return home_zone

        # Count how many zones are interesting
        for zone in self.zones:
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

                elif self.zone_needs_visiting(zone):
                    leavable = True

                # If you can't leave this zone then just forget about it
                if not leavable:
                    continue

                value = self.evaluate_zone(zone, inventory)
                if value:
                    reachable.append((value, zone))

        if not reachable:
            return None

        reachable.sort(key=lambda x: -x[0])
        return reachable[0][1]

class Agent:
    def __init__(self, position, direction, inventory, on_boat, world):
        # Stores the position of the agent
        self.position = position

        # Stores the direction the agent is facing
        self.direction = direction

        # Stores the agent's inventory of held items
        self.inventory = inventory

        # A World object representing the entire scenario
        self.world = world

        # Boolean representing whether the agent is on a boat
        self.on_boat = on_boat

        # The list of preprocessed moves (ie tiles to go through)
        self.moves = []

        # The list of preprocessed actions to take (ie forward, left,
        # right, cut, unlock moves etc.)
        self.actions = []

    # Determine whether the agent has a certain thing in its inventory
    def has(self, thing):
        return thing in self.inventory

    # Given the direction you want to move in, return a list of moves
    # that will get you there
    def get_direction_to_moves(self, way):
        if way == self.direction:
            return ['f']
        elif (way + 1) % 4 == self.direction:
            return ['l', 'f']
        elif (way + 3) % 4 == self.direction:
            return ['r', 'f']
        else:
            return ['r', 'r', 'f']

    # Given a tile adjacent to you, return the absolute direction you
    # must travel in to get there
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

    # Get the moves that you should make on land
    #
    # The overall strategy is as follows:
    #
    # Attempt the first of these strategies you can do:
    # - Get the gold home
    # - Get TO the gold
    # - Get a raft by cutting a tree
    # - Open a door
    # - Get an axe
    # - Get a key
    # - Go somewhere that reveals a bunch of info
    # - Try going across the water with stones or a raft
    # - Cut down a tree even though I have a raft
    # - Pick up a stone
    def get_land_moves(self):
        world = self.world
        path = []
        # Take the gold home
        if '$' in self.inventory:
            # # TODO: Make it so that you can chop trees on the way home
            if self.has('a'):
                path = world.find_nearest(GOTO((WORLD_SIZE//2, WORLD_SIZE//2)), SAFETOTREE, self.position)
            else:
                path = world.find_nearest(GOTO((WORLD_SIZE//2, WORLD_SIZE//2)), SAFETOWALK, self.position)
        # Find the gold
        if not path:
            if self.has('a'):
                path = world.find_nearest(GOLD, SAFETOGOLDWITHAXE, self.position)
            else:
                path = world.find_nearest(GOLD, SAFETOGOLD, self.position)
        # Try cutting a tree
        if not path and self.has('a') and not self.has('w'):
            path = world.find_nearest(TREE, SAFETOWALK, self.position, False)
        # Try unlocking a door
        if not path and self.has('k'):
            path = world.find_nearest(DOOR, SAFETOWALK, self.position, False)
        # Try picking up an axe
        if not path and (not self.has('a')):
            path = world.find_nearest(AXE, SAFETOWALK, self.position, True)
        # Try picking up a key
        if not path and (not self.has('k')):
            path = world.find_nearest(KEY, SAFETOWALK, self.position, True)

        # Try visiting somewhere that will show new info
        if not path:
            # Consider all shortest paths to locations that reveal new stuff
            paths = world.find_nearest_paths(NEEDSVISITING, SAFETOWALK, self.position)
            if paths:
                # Choose the path that reveals the most info using
                # a linear scan
                val = -VISITVALUE(world, paths[0][0])
                best = paths[0]
                for thing in paths:
                    v = -VISITVALUE(world, thing[0])
                    if v < val:
                        best = thing
                path = best

        # At this point we need to ensure that the world has been analysed for next steps
        if not path:
            world.analyse(self.position)
            analysed = True

        # Try determining what to do next in terms of actions over water
        if not path:
            possibly_stone, a = world.get_water_actions(world.get_root(self.position), self.inventory, self.position)
            if a:
                if not possibly_stone:
                    if self.has('w'):
                        # If it must be by boat, go by boat
                        path = world.find_nearest(GOTO(a[-1]), SAFETOWALK, self.position, False)
                    else:
                        # If you don't have a boat, perhaps try getting some stones
                        path = world.find_nearest(STONE, SAFETOSTONE, self.position, False)
                elif self.has('o'):
                    # Place a stone
                    path = world.find_nearest(GOTO(a[-1]), SAFETOSTONE, self.position, False)
                else:
                    # Try doing it with stones
                    if world.n_stones[world.get_root(self.position)] + 1 >= len(a):
                        # Get a stone
                        path = world.find_nearest(STONE, SAFETOSTONE, self.position)
                    else:
                        # Going by boat
                        path = world.find_nearest(GOTO(a[-1]), LAND, self.position, False)

        # If I can't do ANYTHING else, try cutting a tree even though I already have wood
        if not path and self.has('a'):
            path = world.find_nearest(TREE, SAFETOWALK, self.position, False)
        # If I can't do ANYTHING else, try picking up a stone
        if not path:
            path = world.find_nearest(STONE, SAFETOSTONE, self.position, True)

        return path

    # Get the moves that you should make whilst on water
    #
    # The overall strategy is as follows:
    # If you can explore the sea, explore it
    # Otherwise find somewhere to land
    def get_water_moves(self):

        path = []
        # Try visiting everywhere in the water that you can
        paths = self.world.find_nearest_paths(NEEDSVISITING, WATER, self.position)
        if paths:
            # Choose the path that reveals the most info using a
            # linear scan
            val = -VISITVALUE(self.world, paths[0][0])
            best = paths[0]
            for thing in paths:
                v = -VISITVALUE(self.world, thing[0])
                if v < val:
                    best = thing
            path = best
        # Try choosing somewhere to land
        if not path:
            self.world.analyse(self.position)
            target_zone = self.world.choose_landing_zone(self.inventory, self.position)
            path = self.world.find_nearest(lambda w, p: self.world.get_root(p) == target_zone and not TREE(w, p), WATER, self.position, False)

        return path

    # Find a path to take
    # If on a boat call get_water_moves
    # If on land call get_land_moves
    def get_moves(self):
        path = []
        if not self.on_boat:
            path = self.get_land_moves()
        # The case whilst in a boat
        else:
            path = self.get_water_moves()

        return path

    # Get the next action to make
    def get_action(self):
        # If some actions have been preprocessed return one of them
        if self.actions:
            return self.actions.pop(0)

        # If no moves have been preprocessed then find some
        if not self.moves:
            self.moves = self.get_moves()

        # Get the first tile we need to move to
        first = self.moves.pop()

        actions = []

        # Convert them into a list of actions
        way = self.get_direction_to(first)
        actions = self.get_direction_to_moves(way)

        # Change the action if walking into an obstacle
        if TREE(self.world, first):
            actions.remove('f')
            actions.append('c')
            actions.append('f')
        if DOOR(self.world, first):
            actions.remove('f')
            actions.append('u')
            actions.append('f')

        # Save the rest of the actions for later
        self.actions = actions
        return self.actions.pop(0)

    # Convert a relative position in the view field to absolute
    # coordinates in the world
    def view_to_offset(self, x, y):
        if self.direction == 0:
            return (self.position[0] + x, self.position[1] + y)
        elif self.direction == 1:
            return (self.position[0] - y, self.position[1] + x)
        elif self.direction == 2:
            return (self.position[0] - x, self.position[1] - y)
        else:
            return (self.position[0] + y, self.position[1] - x)

    # Based on a given move, determine how this affects the world
    def update_state(self, action):
        # Handle tree cutting
        if action in 'cC' and 'a' in self.inventory:
            # Don't hold duplicate rafts
            if 'w' not in self.inventory:
                self.inventory.append('w')
        # If moving forward
        elif action in 'fF':
            # Calculate position moved to
            self.position = self.view_to_offset(0, -1)
            # Handle leaving the water
            if self.on_boat:
                if not WATER(self.world, self.position):
                    self.on_boat = False
            # Handle picking stuff up
            thing = self.world.access(self.position)
            if thing in 'o$':
                self.inventory.append(thing)
            # No need to track duplicate keys and axes
            if thing in 'ak' and thing not in self.inventory:
                self.inventory.append(self.world.access(self.position))

            # Heading into water
            if self.world.access(self.position) == '~':
                if self.on_boat:
                    pass
                # Handle raft or rock placement
                elif not 'o' in self.inventory:
                    self.on_boat = True
                    self.inventory.remove('w')
                else:
                    self.world.set(self.position, 'O')
                    self.inventory.remove('o')
        # Update direction if turning
        elif action in 'lL':
            self.direction = (self.direction + 3) % 4
        elif action in 'rR':
            self.direction = (self.direction + 1) % 4

    # Print out the world
    def print_world(self):
        self.world.display(self.position, self.direction)
        print("holding [", ", ".join(self.inventory), "]")

# A bunch of helpful lambda functions used to simplify state information
# into slightly simpler yes/no condition

# Returns if a given coordinate is the one we're looking for
GOTO = lambda x: (lambda w, p: p == x)

# A bunch of lambdas expressing whether certain items can be reached
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

# Main function used to run everything
def run_ai():
    moves = 0

    # Agent starts in the middle of the world grid
    origin = (WORLD_SIZE//2, WORLD_SIZE//2)

    # Initialise agent, world objects
    world = World()
    agent = Agent(origin, 0, [], False, world)

    analysed = False
    needs_analysis = False

    # Checks for correct amount of arguments
    if len(sys.argv) != 3:
        print("Usage Python3 " + sys.argv[0] + " -p port \n")
        sys.exit(1)

    port = int(sys.argv[2])

    # Checking for valid port number
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

    world.read_in(sock, agent)
    while 1:
        # Get the next action from the agent
        action = agent.get_action()

        moves += 1

        agent.update_state(action)
        sock.send(action.encode('utf-8'))
        world.read_in(sock, agent)
        agent.print_world()
        print(moves)

    sock.close()

if __name__ == "__main__":
    # Function used to profile the time usage of the AI
    # cProfile.run('run_ai()')

    # Normal function call to run the AI
    run_ai()
