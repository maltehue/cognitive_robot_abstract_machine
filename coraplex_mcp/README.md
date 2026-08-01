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

By default each session starts from a fresh PR2 world. The ``CORAPLEX_MCP_WORLD``
environment variable selects a different belief.

Built-in worlds are selected by name. ``pr2_apartment`` loads a PR2 into the apartment,
built from the PR2 and apartment URDFs with the breakfast objects placed on the counter:

```bash
CORAPLEX_MCP_WORLD=pr2_apartment coraplex-mcp
```

For any other belief, point the variable at a ``module:function`` callable returning a
``World`` or a ``Context`` (build it, deserialize it, or fetch the live twin over ROS):

```python
# my_lab/belief.py
def current_belief():
    return load_my_world()
```

```bash
CORAPLEX_MCP_WORLD=my_lab.belief:current_belief coraplex-mcp
```

Every opened session designs against that belief. Performing a plan deep-copies the
world first, so simulating a design never mutates the belief.

## Response contract

Every tool returns an envelope. A success is ``{"ok": true, "data": ...}``; a failure is
``{"ok": false, "error": {"type", "message", "suggestion"}}``. Tools never raise, so
malformed input is reported to the agent rather than crashing the server.

## Scope and limits

- Performances run against the simulated robot only; the real robot is not driven.
- Operations are serialized so overlapping calls keep the shared execution state
  consistent.
- The number of open sessions is capped (``max_sessions``); opening beyond it fails with
  a ``SessionLimitReached`` error.
- The server logs each tool call under the ``coraplex_mcp`` logger.

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
      "env": {"CORAPLEX_MCP_WORLD": "pr2_apartment"}
    }
  }
}
```

Set ``command``/``args`` so the server runs inside the environment that has CoraPlex and
ROS available. Once connected, ask the agent to open a session, list capabilities, then
perform actions, run plans, or author new capabilities against the belief.
