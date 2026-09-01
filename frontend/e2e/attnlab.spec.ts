import { expect, test } from "@playwright/test";

test("restores a shared comparison and decodes every architecture", async ({
  page,
}) => {
  await page.goto(
    "/?mode=compare&text=one%20two%20three&compare=mha,mqa,mla,kda",
  );

  await expect(
    page.getByRole("heading", { name: "Architecture Compare" }),
  ).toBeVisible();
  await expect(page.locator(".compare-card")).toHaveCount(4);
  await expect(
    page.getByRole("heading", { name: "Multi-Head Attention" }),
  ).toBeVisible();
  const mhaCard = page.locator(".compare-card").first();
  await expect(mhaCard.locator(".compare-metrics dd").first()).toHaveText(
    "192 B",
  );
  await expect(page.getByText("constant", { exact: true })).toBeVisible();

  await page.getByRole("button", { name: "Decode all" }).click();

  await expect(page.locator(".compare-view__actions .phase-badge")).toHaveText(
    "decode",
  );
  await expect(mhaCard.locator(".compare-metrics dd").first()).toHaveText(
    "256 B",
  );
  await expect(page).toHaveURL(/mode=compare/);
  await expect(page).toHaveURL(/compare=mha%2Cmqa%2Cmla%2Ckda/);
});

test("CSA example reaches the end of its trace without blanking", async ({
  page,
}) => {
  const pageErrors: Error[] = [];
  page.on("pageerror", (error) => pageErrors.push(error));
  await page.goto("/");

  await page
    .getByRole("button", { name: /How does CSA choose routes/ })
    .click();
  await expect(
    page.getByRole("heading", { name: "CSA Concept Model" }),
  ).toBeVisible();

  const timeline = page.getByRole("slider", { name: "Execution step" });
  const maximum = await timeline.getAttribute("max");
  await timeline.fill(maximum ?? "0");

  await expect(page.locator(".app-shell")).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Memory View" }),
  ).toBeVisible();
  expect(pageErrors).toEqual([]);
});

test("comparison keeps horizontal overflow inside its panel on mobile", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/?mode=compare&text=one%20two&compare=mha,mqa,mla,kda");
  await expect(page.locator(".compare-card")).toHaveCount(4);

  const dimensions = await page.evaluate(() => ({
    body: document.body.scrollWidth,
    viewport: window.innerWidth,
    panel: document.querySelector(".compare-view")?.scrollWidth ?? 0,
  }));

  expect(dimensions.body).toBeLessThanOrEqual(dimensions.viewport);
  expect(dimensions.panel).toBeGreaterThan(dimensions.viewport);
});
