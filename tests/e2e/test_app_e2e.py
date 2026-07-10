"""
Playwright E2E tests for the SPP Weis Nodal Price Forecast Shiny app.

These tests use a lightweight test app (app_for_test.py) that patches
heavy startup loaders so no R2/S3 credentials are needed.
"""

import re

from playwright.sync_api import Page, expect


def test_app_title(page: Page, app):
    """Page title matches the expected value."""
    page.goto(app.url)
    expect(page).to_have_title("SPP IM West Nodal Price Forecast")


def test_sidebar_inputs_render(page: Page, app):
    """Date, hour, node, and n_days selects are visible after data loads."""
    page.goto(app.url)

    # Wait for data to load — node_name select gets populated
    node_select = page.locator("#node_name")
    node_select.wait_for(state="visible", timeout=30_000)

    # Wait for choices to be populated (not just the empty placeholder)
    page.wait_for_function(
        "document.querySelector('#node_name option') !== null"
        " && document.querySelector('#node_name').options.length > 0",
        timeout=30_000,
    )

    expect(page.locator("#fcast_date")).to_be_visible()
    expect(page.locator("#fcast_hour")).to_be_visible()
    expect(page.locator("#node_name")).to_be_visible()
    expect(page.locator("#n_days")).to_be_visible()


def test_data_updated_text(page: Page, app):
    """Sidebar contains '4 hours' note about data refresh cadence."""
    page.goto(app.url)
    page.wait_for_timeout(2_000)
    sidebar = page.locator(".bslib-sidebar-layout .sidebar")
    expect(sidebar).to_contain_text("4 hours")


def test_forecast_generates(page: Page, app):
    """Clicking 'Get forecast' produces a plot and data section."""
    page.goto(app.url)

    # Wait for inputs to be populated
    page.wait_for_function(
        "document.querySelector('#node_name') !== null"
        " && document.querySelector('#node_name').options.length > 0",
        timeout=30_000,
    )

    # Click forecast button
    page.locator("#get_fcast_btn").click()

    # Wait for forecast header to appear (contains node name)
    forecast_header = page.locator("#forecast_header")
    expect(forecast_header).not_to_be_empty(timeout=60_000)

    # Forecast data section should also appear
    forecast_data = page.locator("#forecast_data_section")
    expect(forecast_data).not_to_be_empty(timeout=60_000)


def test_stale_forecast_clears(page: Page, app):
    """Changing a forecast input after forecast clears the output."""
    page.goto(app.url)

    # Wait for inputs
    page.wait_for_function(
        "document.querySelector('#node_name') !== null"
        " && document.querySelector('#node_name').options.length > 0",
        timeout=30_000,
    )

    # Generate forecast
    page.locator("#get_fcast_btn").click()
    forecast_header = page.locator("#forecast_header")
    expect(forecast_header).not_to_be_empty(timeout=60_000)

    # Change n_days to trigger _clear_stale_forecast
    page.select_option("#n_days", "3")

    # Forecast placeholder should re-appear (empty header = cleared)
    placeholder = page.locator("#forecast_placeholder")
    expect(placeholder).not_to_be_empty(timeout=15_000)


def test_market_toggle_reloads_and_forecasts(page: Page, app):
    """Switching the market (day-ahead -> real-time) reloads the model and forecasts."""
    page.goto(app.url)
    page.wait_for_function(
        "document.querySelector('#node_name') !== null"
        " && document.querySelector('#node_name').options.length > 0",
        timeout=30_000,
    )
    # Day-ahead is the default market; capture its loaded-model timestamp.
    expect(page.locator("#target")).to_have_value("da")
    da_ts = page.locator("#train_timestamp_display").inner_text().strip()

    # Switch to real-time; the app reloads that target's data + champion model.
    page.select_option("#target", "rt")
    expect(page.locator("#target")).to_have_value("rt")

    # The RT champion has a different train timestamp than DA, so a change in the
    # displayed timestamp signals the async reload finished — wait for it before
    # forecasting (otherwise the reload's dropdown refresh clears a stale click).
    page.wait_for_function(
        "prev => { const el = document.querySelector('#train_timestamp_display');"
        " return el && el.innerText.trim() !== '' && el.innerText.trim() !== prev; }",
        arg=da_ts,
        timeout=90_000,
    )
    # Let the post-reload dropdown refresh settle before clicking.
    page.wait_for_timeout(2_000)

    # A forecast on the RT model renders, and the header names the market.
    page.locator("#get_fcast_btn").click()
    forecast_header = page.locator("#forecast_header")
    expect(forecast_header).not_to_be_empty(timeout=60_000)
    expect(forecast_header).to_contain_text("Real-time")


def test_download_filename(page: Page, app):
    """Download link has clean timestamp format (no spaces or colons)."""
    page.goto(app.url)

    # Wait for inputs
    page.wait_for_function(
        "document.querySelector('#node_name') !== null"
        " && document.querySelector('#node_name').options.length > 0",
        timeout=30_000,
    )

    # Generate forecast
    page.locator("#get_fcast_btn").click()

    # Wait for download button to appear inside forecast_data_section
    download_btn = page.locator("#forecast_data_section #download_data")
    expect(download_btn).to_be_visible(timeout=60_000)

    # Start a download
    with page.expect_download(timeout=30_000) as download_info:
        download_btn.click()
    download = download_info.value
    filename = download.suggested_filename

    # Filename should match pattern: price-forecast-<NODE>-<YYYY-MM-DDTHH-MM>.csv
    assert filename.endswith(".csv"), f"Expected .csv, got {filename}"
    assert " " not in filename, f"Filename contains spaces: {filename}"
    assert ":" not in filename, f"Filename contains colons: {filename}"
    assert re.match(r"price-forecast-.+-\d{4}-\d{2}-\d{2}T\d{2}-\d{2}\.csv", filename), \
        f"Unexpected filename format: {filename}"
