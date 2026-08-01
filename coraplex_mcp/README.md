# CoraPlex MCP

A [Model Context Protocol](https://modelcontextprotocol.io) server that exposes the
CoraPlex robot control architecture to language-model agents, so an agent can both
*compose* robot control programs from existing capabilities and *author* new
perception and manipulation capabilities that slot into the same plans.

## Architecture

The package separates the transport-independent domain logic from the MCP wire
protocol, so the domain logic is testable without the `mcp` dependency:

- ``catalogue`` - reflects the CoraPlex action and motion classes into serializable
  schemas an agent can discover.
- ``marshaling`` - converts JSON tool arguments into CoraPlex value types (poses,
  bodies, enums) against a live world.
- ``authoring`` - synthesizes new ``ActionDescription`` subclasses from a declarative
  specification, so authored capabilities are ordinary actions that flow into a plan
  exactly like the built-ins.
- ``sessions`` - holds the ``Context`` (world plus robot) a program is built against.
- ``validation`` - test-drives a capability or plan in the simulated (belief-state)
  robot and reports its outcome.
- ``server`` - the thin FastMCP binding that turns the above into tools and resources.

## Running the server

```bash
coraplex-mcp
```

The server defaults to the simulated robot; the real-robot path requires a running
ROS 2 environment and is opt-in.
