/**
 * Formatting helper that outlived the catalogue.
 *
 * This file used to browse the HuggingFace model catalogue — `browseModelCards`
 * and `listModelCategories`, both of which reached the Hub. This build reaches
 * no catalogue, so they are gone and only the byte formatter remains, which the
 * Models screen uses for weights already on disk.
 */

/** Bytes as GB with one decimal, e.g. `4.7 GB`. */
export function formatSize(bytes: number): string {
  const gb = bytes / 1024 ** 3;
  if (gb >= 1) return `${gb.toFixed(1)} GB`;
  return `${Math.round(bytes / 1024 ** 2)} MB`;
}
