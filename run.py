import solidify
import gen_graph
import find_shortest_path
import guide_bot
import time

# solidify the non-path region by converting to contrasting color
solidify.main()

# separate the path from the solidified image, find the skeleton, convert to graph
gen_graph.main()

# find shortest path between the starting and ending point of the graph
find_shortest_path.main()

# plot the path on the image
guide_bot.main()

if 0xFF == ord('q'):
    quit
