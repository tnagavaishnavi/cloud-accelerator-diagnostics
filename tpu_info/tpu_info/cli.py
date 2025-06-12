# Copyright 2023 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#      https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines command line interface for `tpu-info` tool.

Top-level functions should be added to `project.scripts` in `pyproject.toml`.
"""

from datetime import datetime
import sys
import time
from typing import Any, List

from tpu_info import args
from tpu_info import device
from tpu_info import metrics
import grpc
from rich.console import Console, Group
from rich.console import RenderableType
from rich.live import Live
from rich.panel import Panel
import rich.table

def _bytes_to_gib(size: int) -> float:
  return size / (1 << 30)


# TODO(vidishasethi): b/418938764 - Modularize by extracting
#  each table's rendering logic into its own dedicated helper function.
def _fetch_and_render_tables(chip_type: Any, count: int)-> List[RenderableType]:
  """Fetches all TPU data and prepares a list of Rich Table objects for display."""
  renderables: List[RenderableType] = []

  table = rich.table.Table(title="TPU Chips", title_justify="left")
  table.add_column("Chip")
  table.add_column("Type")
  table.add_column("Devices")
  # TODO(wcromar): this may not match the libtpu runtime metrics
  # table.add_column("HBM (per core)")
  table.add_column("PID")

  chip_paths = [device.chip_path(chip_type, index) for index in range(count)]
  chip_owners = device.get_chip_owners()

  for chip in chip_paths:
    owner = chip_owners.get(chip)

    table.add_row(
        chip,
        str(chip_type),
        str(chip_type.value.devices_per_chip),
        str(owner),
    )

  renderables.append(table)

  table = rich.table.Table(
      title="TPU Runtime Utilization", title_justify="left"
  )
  table.add_column("Device")
  table.add_column("HBM usage")
  table.add_column("Duty cycle", justify="right")

  try:
    device_usage = metrics.get_chip_usage(chip_type)
  except grpc.RpcError as e:
    if e.code() == grpc.StatusCode.UNAVAILABLE:  # pytype: disable=attribute-error
      error_message = (
          "Libtpu metrics unavailable. Is there a framework using the"
          " TPU? See"
          " [link=https://github.com/google/cloud-accelerator-diagnostics/"
          "tree/main/tpu_info]tpu_info docs[/link]"
          " for more information."
      )
      error_message_renderable = Panel(
          f"[yellow]WARNING:[/yellow] {error_message}",
          title="[b]Runtime Utilization Status[/b]",
          border_style="yellow",
      )
    else:
      error_message = f"ERROR fetching runtime utilization: {e}"
      error_message_renderable = Panel(f"[red]{error_message}[/red]",
                                       title="[b]Runtime Utilization Error[/b]",
                                       border_style="red")
    renderables.append(error_message_renderable)

    device_usage = [metrics.Usage(i, -1, -1, -1) for i in range(count)]

  # TODO(wcromar): take alternative ports as a flag
  # print("Connected to libtpu at grpc://localhost:8431...")
  for chip in device_usage:
    if chip.memory_usage < 0:
      memory_usage = "N/A"
    else:
      memory_usage = (
          f"{_bytes_to_gib(chip.memory_usage):.2f} GiB /"
          f" {_bytes_to_gib(chip.total_memory):.2f} GiB"
      )
    if chip.duty_cycle_pct < 0:
      duty_cycle_pct = "N/A"
    else:
      duty_cycle_pct = f"{chip.duty_cycle_pct:.2f}%"
    table.add_row(
        str(chip.device_id),
        memory_usage,
        duty_cycle_pct
        if chip_type.value.devices_per_chip == 1 or chip.device_id % 2 == 0
        else "",
    )

  renderables.append(table)

  table = rich.table.Table(title="TensorCore Utilization", title_justify="left")
  table.add_column("Chip ID")
  table.add_column("TensorCore Utilization", justify="right")

  try:
    # pylint: disable=g-import-not-at-top
    from libtpu import sdk  # pytype: disable=import-error

    tensorcore_util_data = sdk.monitoring.get_metric("tensorcore_util").data()
  except ImportError as e:
    renderables.append(
        Panel(
            f"[yellow]WARNING: ImportError: {e}. libtpu SDK not available.[/]",
            title="[b]TensorCore Status[/b]",
            border_style="yellow",
        )
    )
  except AttributeError as e:
    renderables.append(
        Panel(
            f"[yellow]WARNING: AttributeError: {e}. Please check if the"
            " latest libtpu is used.[/]",
            title="[b]TensorCore Status[/b]",
            border_style="yellow",
        )
    )
  except RuntimeError as e:
    renderables.append(
        Panel(
            f"[red]ERROR: RuntimeError: {e}. Please check if the latest vbar"
            " control agent is used.[/]",
            title="[b]TensorCore Status[/b]",
            border_style="red",
        )
    )
  else:
    for i in range(len(tensorcore_util_data)):
      tc_data = f"{tensorcore_util_data[i]}%"
      table.add_row(
          str(i),
          tc_data,
      )
    renderables.append(table)

  table = rich.table.Table(
      title="TPU Buffer Transfer Latency", title_justify="left"
  )
  table.add_column("Buffer Size")
  table.add_column("P50", justify="right")
  table.add_column("P90", justify="right")
  table.add_column("P95", justify="right")
  table.add_column("P999", justify="right")

  try:
    buffer_transfer_latency_distributions = (
        metrics.get_buffer_transfer_latency()
    )
  except grpc.RpcError as e:
    if e.code() == grpc.StatusCode.UNAVAILABLE:  # pytype: disable=attribute-error
      error_message = (
          "Buffer Transfer Latency metrics unavailable. Did you start"
          " a MULTI_SLICE workload with"
          " `TPU_RUNTIME_METRICS_PORTS=8431,8432,8433,8434`?"
      )
      renderables.append(
          Panel(f"[yellow]WARNING:[/yellow] {error_message}",
                title="[b]Buffer Transfer Latency Status[/b]",
                border_style="yellow")
      )

    else:
      error_message = f"ERROR fetching buffer transfer latency: {e}"
      renderables.append(
          Panel(f"[red]{error_message}[/red]",
                title="[b]Buffer Transfer Latency Error[/b]",
                border_style="red")
      )

    buffer_transfer_latency_distributions = []

  for distribution in buffer_transfer_latency_distributions:
    table.add_row(
        distribution.buffer_size,
        f"{distribution.p50:.2f} us",
        f"{distribution.p90:.2f} us",
        f"{distribution.p95:.2f} us",
        f"{distribution.p999:.2f} us",
    )
  renderables.append(table)

  return renderables


def _get_runtime_info(rate: float) -> Panel:
  """Returns a Rich Panel with runtime info for the streaming mode."""
  current_ts = time.time()
  last_updated_time_str = datetime.fromtimestamp(current_ts).strftime(
      "%H:%M:%S %Y-%m-%d"
  )
  runtime_status = (
      f"Refresh rate: {rate}s | Last update: {last_updated_time_str}"
  )
  status_panel = Panel(
      runtime_status, title="[b]Streaming Status[/b]", border_style="green"
  )
  return status_panel


def print_chip_info():
  """Print local TPU devices and libtpu runtime metrics."""
  cli_args = args.parse_arguments()
  # TODO(wcromar): Merge all of this info into one table
  chip_type, count = device.get_local_chips()
  if not chip_type:
    print("No TPU chips found.")
    return

  if cli_args.streaming:
    if cli_args.rate <= 0:
      print("Error: Refresh rate must be positive.", file=sys.stderr)
      return
    print(
        f"Starting streaming mode (refresh rate: {cli_args.rate}s). Press"
        " Ctrl+C to exit."
    )

    if cli_args.rate > 0:
      data_refresh_hz = 1.0 / cli_args.rate
      target_screen_fps = data_refresh_hz * 1.2
      screen_refresh_per_second = min(max(4, int(target_screen_fps)), 30)
    else:
      screen_refresh_per_second = 4
    try:
      renderables = _fetch_and_render_tables(chip_type, count)
      streaming_status = _get_runtime_info(cli_args.rate)

      if not renderables and chip_type:
        print(
            "No data tables could be generated. Exiting streaming.",
            file=sys.stderr,
        )
        return

      render_group = Group(*(renderables if renderables else []))

      with Live(
          streaming_status,
          render_group,
          refresh_per_second=screen_refresh_per_second,
          screen=True,
          vertical_overflow="visible",
      ) as live:
        while True:
          try:
            time.sleep(cli_args.rate)
            new_renderables = _fetch_and_render_tables(chip_type, count)
            render_group = Group(*(new_renderables if new_renderables else []))
            streaming_status = _get_runtime_info(cli_args.rate)

            live.update(streaming_status, render_group)
          except Exception as loop_e:
            import traceback
            print(f"\nFATAL ERROR during streaming update cycle, stopping stream: {type(loop_e).__name__}: {loop_e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            raise loop_e 
    except KeyboardInterrupt:
      print("\nExiting streaming mode.")
    except Exception as e:
      import traceback

      print(
          f"\nAn unexpected error occurred in streaming mode: {e}",
          file=sys.stderr,
      )
      traceback.print_exc(file=sys.stderr)
      sys.exit(1)

  else:
    renderables = _fetch_and_render_tables(chip_type, count)

    if renderables:
      console_obj = Console()
      for item in renderables:
        console_obj.print(item)
