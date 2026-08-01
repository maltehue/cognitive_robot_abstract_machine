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

The server speaks the Model Context Protocol over standard input and output, so an MCP
client launches it as a subprocess. It must run where the CoraPlex and ROS stack is
installed, because building and simulating a world uses the ROS toolchain.

## Designing against an existing belief

By default each session starts from a fresh PR2 world. To design against an existing
semantic digital twin instead, point the server at a callable that returns the belief
through the ``CORAPLEX_MCP_WORLD`` environment variable, written as ``module:function``.
The callable may return a ``World`` or a ``Context``:

```python
# my_lab/belief.py
def current_belief():
    world = load_my_world()  # build, deserialize, or fetch the live twin over ROS
    return world
```

```bash
CORAPLEX_MCP_WORLD=my_lab.belief:current_belief coraplex-mcp
```

Every opened session designs against that belief. Performing a plan deep-copies the
world first, so simulating a design never mutates the belief.

## Connecting a client

Register the server with an MCP client such as Claude Desktop (in its
``claude_desktop_config.json``) or Claude Code (``claude mcp add``). A Claude Desktop
entry looks like:

```json
{
  "mcpServers": {
    "coraplex-robot-control": {
      "command": "uv",
      "args": ["run", "coraplex-mcp"],
      "env": {"CORAPLEX_MCP_WORLD": "my_lab.belief:current_belief"}
    }
  }
}
```

Set ``command``/``args`` so the server runs inside the environment that has CoraPlex and
ROS available. Once connected, ask the agent to open a session, list capabilities, then
perform actions, run plans, or author new capabilities against the belief.
