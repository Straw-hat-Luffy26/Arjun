/**
 * What each tool is, in the words the model actually decides from.
 *
 * ## Why the descriptions are this long
 *
 * A 7B model picks a tool from these sentences and nothing else. It has no
 * source, no documentation and no memory of the last run. Every characteristic
 * failure this product has seen is addressable in prose, and each description
 * below says the thing that prevents its own tool's mistake:
 *
 * - answering from memory instead of searching
 * - validating a document it never produced
 * - writing a deliverable with the plain-text tool
 * - narrating the output of code that never ran
 * - quoting a passage it was only shown the citation for
 * - treating a truncated result as a complete one
 *
 * Six clauses appear in every description, in the same order, because a model
 * reading a familiar shape finds the clause it needs rather than re-reading the
 * paragraph: what it does, when to use it, when not to, what it changes, what
 * it costs, and what to do when it fails. The repetition is the feature.
 *
 * ## Why the names carry a namespace
 *
 * `artifact.verify_docx` is visibly the partner of
 * `artifact.create_approval_note` and visibly not a way to read a file. The old
 * flat names — `validate_artifact`, `read_scoped_file` — read as unrelated
 * verbs, and the characteristic error was reaching for a near neighbour. The
 * authority for these names is `ToolName` in
 * `src-tauri/src/orchestrator/tools.rs`; a name absent there is refused by the
 * gateway however it is spelled here.
 *
 * ## Why `readOnly` lives here rather than being decided per tool
 *
 * It is one field, and everything else follows from it: whether the call may run
 * beside another, whether a person is interrupted, whether a resumption must
 * avoid repeating it. Writing it once per tool and deriving the rest means a
 * tool cannot be read-only in one place and side-effecting in another.
 */

import { Type, type TObject } from "typebox";

/** One tool, before it is bound to a peer. */
export interface ToolDefinition {
  /** The wire name. Must exist in Rust's `ToolName`. */
  name: string;
  /** What a person sees in the run trace. */
  label: string;
  description: string;
  parameters: TObject;
  /**
   * Whether the call only reads.
   *
   * Mirrors `ToolName::is_read_only` in Rust, which is the authority. Held here
   * too because the runtime decides execution mode before Rust is asked
   * anything — and checked against Rust's answer in the tests, so the two
   * cannot drift into disagreeing about which calls may overlap.
   */
  readOnly: boolean;
}

/**
 * Schemas are closed.
 *
 * `additionalProperties: false` turns a hallucinated argument into a validation
 * failure the model reads and corrects, rather than a silently ignored field
 * that makes the call do something subtly different from what it asked for. A
 * model that writes `pages: "4-6"` alongside `fromPage` should be told, not
 * quietly given page 1.
 */
function closed(properties: Parameters<typeof Type.Object>[0]): TObject {
  return Type.Object(properties, { additionalProperties: false });
}

/** Every tool this runtime knows how to build, keyed by wire name. */
export const TOOL_DEFINITIONS: readonly ToolDefinition[] = [
  {
    name: "knowledge.search_authorized",
    label: "Search documents",
    readOnly: true,
    description:
      "Searches the organisation's indexed documents and returns passages with their source and " +
      "page, each marked [E1], [E2] and so on for citation. " +
      "Use it before answering anything about this organisation's own procedure, specification, " +
      "correspondence or figures — and use it first, before reasoning about what the answer " +
      "might be. Do not use it for greetings, small talk, general knowledge, mathematics or " +
      "writing code: none of those live in the collections, and searching for them wastes a " +
      "turn and produces a refusal where an answer belonged. " +
      "Do not answer such questions from memory, and do not treat an empty result as proof a thing is " +
      "false: results are filtered by what the signed-in person is permitted to read, so a document " +
      "may exist and be out of scope. " +
      "Effects: none. It only reads, needs nobody's approval, and touches no network. " +
      "Limits: at most 6 passages per call, and a long result is cut deterministically with a " +
      "line saying so — the same input always produces the same cut, byte for byte. " +
      'Set detail to "citations" to see only sources and pages, which is much cheaper when you ' +
      "are deciding which passages you want. Use page to fetch the next batch when a result " +
      "says it was truncated. " +
      "If it finds nothing: search again with the specific technical term rather than a " +
      "paraphrase, and if that also finds nothing say no source was found rather than " +
      "answering the document question anyway. That applies to the document question you " +
      "searched for, not to the rest of the conversation.",
    parameters: closed({
      query: Type.String({
        description:
          "What to look for, in natural language. Specific technical terms retrieve better than paraphrase.",
        minLength: 1,
      }),
      detail: Type.Optional(
        Type.Union([Type.Literal("passages"), Type.Literal("citations")], {
          description:
            'How much to return. "passages" (the default) returns the text; "citations" returns ' +
            "only the source and page of each, which you cannot quote from.",
        }),
      ),
      maxResults: Type.Optional(
        Type.Integer({
          minimum: 1,
          maximum: 6,
          description: "How many passages to return. Defaults to 6.",
        }),
      ),
      page: Type.Optional(
        Type.Integer({
          minimum: 1,
          description:
            "Which page of results to return, 1-based. Defaults to 1. Use when a previous " +
            "call said its result was truncated and you need the next batch.",
        }),
      ),
    }),
  },

  {
    name: "knowledge.load_evidence_region",
    label: "Read more of a document",
    readOnly: true,
    description:
      "Reads a named page range of a document you have already retrieved a passage from, and adds " +
      "those passages to this task's evidence. " +
      "Use it when a passage stops mid-clause, a table continues overleaf, or you need the " +
      "paragraph around a citation. " +
      "Do not use it to read a whole document, and do not use it to reach a document you have " +
      "not searched: the documentSha256 comes from a passage a search already returned. " +
      "Effects: none. It only reads, needs nobody's approval, and touches no network. " +
      "Limits: at most 10 pages per call; a wider range is refused rather than quietly trimmed, " +
      "so you always know what you did and did not get. Truncation is deterministic — the same " +
      "input always produces the same cut. " +
      "If it is refused for being too wide: ask for the few pages you actually need. If a page " +
      "comes back empty it may be a scan — use media.extract_findings to find out.",
    parameters: closed({
      documentSha256: Type.String({
        description: "The document identifier carried on a passage you already retrieved.",
        minLength: 1,
      }),
      fromPage: Type.Integer({ minimum: 1, description: "First page to read, inclusive." }),
      toPage: Type.Optional(
        Type.Integer({
          minimum: 1,
          description: "Last page to read, inclusive. Defaults to fromPage. At most 10 pages.",
        }),
      ),
    }),
  },

  {
    name: "media.extract_findings",
    label: "Read a scanned page range",
    readOnly: true,
    description:
      "Reports what a page range of a document does and does not yield, separating pages with " +
      "extracted text from pages that are images nobody has read. " +
      "Use it when a page range came back empty or thin and you need to know whether the clause " +
      "is genuinely absent or simply unread — those two lead to opposite conclusions and this is " +
      "the only tool that tells them apart. " +
      "Do not use it as a general reader: knowledge.load_evidence_region is for pages that have " +
      "text, and this one exists for the pages that do not. " +
      "Effects: none. It only reads. It talks to a local extraction sidecar on this machine and " +
      "reaches no outside network. " +
      "Limits: the same 10-page range as load_evidence_region. This deployment may have no OCR or " +
      "vision model installed, in which case it says so. " +
      "If it reports pages unread: say the pages could not be read and that a person needs to look " +
      "at them. Never describe or quote a page reported as unread.",
    parameters: closed({
      documentSha256: Type.String({
        description: "The document identifier carried on a passage you already retrieved.",
        minLength: 1,
      }),
      fromPage: Type.Integer({ minimum: 1, description: "First page to examine, inclusive." }),
      toPage: Type.Optional(
        Type.Integer({
          minimum: 1,
          description: "Last page to examine, inclusive. Defaults to fromPage. At most 10 pages.",
        }),
      ),
    }),
  },

  {
    name: "document.read_pages",
    label: "Read more of an attached document",
    readOnly: true,
    description:
      "Reads a page range of a document attached to THIS conversation, by the id on its " +
      "<attachment> tag. " +
      "Use it whenever an attachment says only part of it was included, or that none of it was, " +
      "or when you need a page you have not been shown — including pages of a document attached " +
      "in an earlier turn. Every page of every attachment was read and stored before this turn " +
      "started, so a page missing from the prompt is not a page that is unavailable. " +
      "Do not use it for documents from the knowledge base: those are reached with " +
      "knowledge.load_evidence_region. This one reads only what the person attached to this " +
      "conversation. " +
      "Effects: none. It reads a store on this machine, uses no model, and touches no network. " +
      "Limits: at most 10 pages per call; a wider range is refused rather than quietly trimmed, " +
      "so you always know what you did and did not get. A long range may also stop early on size, " +
      "and says which pages it did not return. " +
      "If it reports a page as not returned: ask for it in a narrower range, or say plainly that " +
      "it could not be read. Never describe or quote a page that was not returned to you.",
    parameters: closed({
      documentSha256: Type.String({
        description:
          "The id on the document's <attachment> tag in this conversation. Not a file name.",
        minLength: 1,
      }),
      fromPage: Type.Integer({ minimum: 1, description: "First page to read, inclusive." }),
      toPage: Type.Optional(
        Type.Integer({
          minimum: 1,
          description: "Last page to read, inclusive. Defaults to fromPage. At most 10 pages.",
        }),
      ),
    }),
  },

  {
    name: "document.search",
    label: "Find a passage in an attached document",
    readOnly: true,
    description:
      "Finds passages by what they say, across every document attached to THIS conversation. " +
      "Use it whenever an attachment says only part of it is shown, or when you need something " +
      "you have not been given and do not know which page holds it — which is the usual case. " +
      "Every page of every attachment was read and stored before this turn started, so a " +
      "passage missing from the prompt is not a passage that is unavailable. Prefer it over " +
      "document.read_pages unless you already know the page number: guessing page ranges costs " +
      "several calls and this costs one. " +
      "Do not use it for the knowledge base: those documents are reached with " +
      "knowledge.search_authorized. This one searches only what the person attached here. " +
      "Effects: none. It reads a store on this machine, uses no model, and touches no network. " +
      "Limits: at most 6 passages per call, and a large result may stop early on size and say " +
      "so. Each passage comes back with its document, page and heading trail — quote those " +
      "when you use it. " +
      "If it finds nothing, it says how many sections it looked at: try different words, or " +
      "read a page range with document.read_pages. Never describe a passage you were not shown.",
    parameters: closed({
      query: Type.String({
        description:
          "The words you expect to find — a tag number, a clause title, a part name, a unit.",
        minLength: 1,
      }),
    }),
  },
  {
    name: "notebook.list",
    label: "List the notebooks",
    readOnly: true,
    description:
      "Names every notebook the signed-in person has, with how many documents each holds. " +
      "Use it when they ask what notebooks exist, or when you need the exact name of one " +
      "before acting on it. " +
      "Do not use it to read what is inside a notebook - that is notebook.list_sources for " +
      "the file names, and document.search for what they say. " +
      "Effects: none, a single read of a local database. " +
      "Limits: names and counts only; it never returns document contents. " +
      "If it says there are none, say so and offer to create one.",
    parameters: closed({}),
  },
  {
    name: "notebook.create",
    label: "Create a notebook",
    readOnly: false,
    description:
      "Makes a new, empty notebook with the name given. " +
      "Use it when the person asks for a notebook to be created. " +
      "Do not use it to add documents - a new notebook is empty, and notebook.add_source " +
      "puts an attached file into it afterwards. Do not create one to hold an answer or a " +
      "note; a notebook groups source documents, it is not a scratchpad, and " +
      "workspace.write_text is the tool for writing a file. " +
      "Effects: one row in a local database. Nothing is downloaded and no file is written. " +
      "Limits: names are at most 120 characters, and a name already in use is refused rather " +
      "than duplicated. " +
      "If it says the name is taken, use the existing notebook instead of inventing a variant.",
    parameters: closed({
      name: Type.String({
        description: "What to call it, in the person's own words.",
        minLength: 1,
      }),
    }),
  },
  {
    name: "notebook.rename",
    label: "Rename a notebook",
    readOnly: false,
    description:
      "Changes a notebook's name, keeping its documents and its graph. " +
      "Use it when the person asks for one to be renamed or called something else. " +
      "Do not use it to make a copy - there is one notebook before and after, under a new " +
      "name. " +
      "Effects: one row updated in a local database. " +
      "Limits: the same 120-character limit as creating one. " +
      "If it cannot tell which notebook is meant, it says which exist; ask rather than guess.",
    parameters: closed({
      notebook: Type.Optional(
        Type.String({
          description:
            "Which notebook, by name or part of one. Omit it only when they have a single " +
            "notebook.",
        }),
      ),
      name: Type.String({ description: "The new name.", minLength: 1 }),
    }),
  },
  {
    name: "notebook.delete",
    label: "Delete a notebook",
    readOnly: false,
    description:
      "Deletes a notebook, its list of sources, and the whole knowledge graph built over it. " +
      "Use it only when the person has clearly asked for that notebook to be deleted. " +
      "Do not use it to clear a notebook out so it can be refilled - notebook.remove_source " +
      "takes single sources out and keeps the notebook. Do not use it to tidy up on your own " +
      "initiative. " +
      "Effects: rows removed from five local tables, and the graph cannot be recovered without " +
      "running the extraction passes again. The documents themselves are NOT deleted and stay " +
      "attached wherever they were. This asks the person before it runs. " +
      "Limits: one notebook per call, and only one belonging to the signed-in person. " +
      "If it cannot tell which notebook is meant, it lists them rather than choosing; ask. " +
      "If the person has not clearly said to delete it, do not call this - confirm first.",
    parameters: closed({
      notebook: Type.Optional(
        Type.String({
          description: "Which notebook, by name. Omit only when they have a single notebook.",
        }),
      ),
    }),
  },
  {
    name: "notebook.list_sources",
    label: "List the sources in a notebook",
    readOnly: true,
    description:
      "Names the documents in a notebook, with the content hash of each. " +
      "Use it when the person asks what is in a notebook, and before removing a source so " +
      "you know what is there. The hash it returns is what document.read_pages takes. " +
      "Do not use it to read the documents - it returns names, never text. " +
      "Effects: none, a single read of a local database. " +
      "Limits: file names and hashes only. " +
      "If it says the notebook is empty, say so rather than describing what might be in it.",
    parameters: closed({
      notebook: Type.Optional(
        Type.String({
          description: "Which notebook, by name. Omit only when they have a single notebook.",
        }),
      ),
    }),
  },
  {
    name: "notebook.add_source",
    label: "Add an attached document to a notebook",
    readOnly: false,
    description:
      "Puts a document attached to THIS conversation into a notebook. " +
      "Use it when the person attaches a file and asks for it to go into one of their " +
      "notebooks. " +
      "Do not use it for a file that is not attached here - it can only reach this " +
      "conversation's own attachments, and naming anything else is refused. Adding a document " +
      "does not put it in the graph; Build graph on the Notebooks screen does that. " +
      "Effects: one row in a local database. The document is not copied or moved. " +
      "Limits: one document per call, named exactly enough to be unambiguous. " +
      "If it says nothing of that name is attached, list what is rather than guessing.",
    parameters: closed({
      notebook: Type.Optional(
        Type.String({
          description: "Which notebook, by name. Omit only when they have a single notebook.",
        }),
      ),
      document: Type.String({
        description: "The attached document's file name, as shown in this conversation.",
        minLength: 1,
      }),
    }),
  },
  {
    name: "notebook.remove_source",
    label: "Take a source out of a notebook",
    readOnly: false,
    description:
      "Removes one document from a notebook, along with the graph evidence that came from it. " +
      "Use it when the person asks for a source to be taken out. " +
      "Do not use it to delete the document - the file itself is untouched and stays attached " +
      "where it was; this only takes it out of this notebook. " +
      "Effects: rows removed from the notebook's membership and evidence tables. The terms " +
      "that came only from this document stop being cited. " +
      "Limits: one document per call. " +
      "If it says the name matches more than one source, name it exactly rather than " +
      "picking one.",
    parameters: closed({
      notebook: Type.Optional(
        Type.String({
          description: "Which notebook, by name. Omit only when they have a single notebook.",
        }),
      ),
      document: Type.String({
        description: "The source's file name, as notebook.list_sources shows it.",
        minLength: 1,
      }),
    }),
  },
  {
    name: "knowledge.build_graph",
    label: "Draw the knowledge graph of a notebook",
    readOnly: true,
    description:
      "Returns the knowledge graph of a notebook as a Mermaid diagram: the things its " +
      "documents name, and the links between them. Use it whenever the person asks to see " +
      "how something connects - a diagram, a map, a chart of suppliers and equipment, " +
      "'how does X relate to Y', 'show me the structure'. Prefer it over describing a " +
      "structure in prose: a list of pairs is the worst way to answer a question about shape. " +
      "Pass documentSha256 to draw one file rather than the whole notebook, and focus to " +
      "centre the picture on one term. " +
      "Do not use it to find out what a document says - that is document.search. This " +
      "draws the shape of what was already found, and it cannot quote or explain a " +
      "passage. " +
      "You do not need a notebook id: pass the name the person used, or nothing at all. " +
      "The notebooks available are listed in your context. " +
      "It READS a graph that was already built; it does not build one. If it says the " +
      "notebook has no graph yet, say so and tell the person to run Build graph on the " +
      "Notebooks screen - do not retry, and do not draw a diagram from your own reading of " +
      "the documents. " +
      "Effects: none. Three reads of a local database; no model is started and no network " +
      "is touched. " +
      "Limits: at most 60 links are drawn and the diagram says when more were left out. " +
      "A solid arrow is a relation actually read out of the documents. A dashed line means " +
      "only that two terms appear in the same passage - report those as co-occurrence, " +
      "never as a relationship the documents stated.",
    parameters: closed({
      notebook: Type.Optional(
        Type.String({
          description:
            "Which notebook, in the person's own words - its name, or part of it. " +
            "Omit it when they did not name one: if they have a single notebook it is " +
            "used, and if they have several the tool answers with their names so you " +
            "can ask. Never invent an id.",
        }),
      ),
      documentSha256: Type.Optional(
        Type.String({
          description:
            "Narrow the whole graph, counts included, to one document of that notebook.",
        }),
      ),
      focus: Type.Optional(
        Type.String({
          description:
            "A node id to centre on. The picture then shows its neighbourhood rather than " +
            "the whole graph.",
        }),
      ),
    }),
  },

  {
    name: "knowledge.multimodal_retrieve",
    label: "Search text, image regions, and tables",
    readOnly: true,
    description:
      "Searches the text index, the image-region index, and the table index together, and returns " +
      "matching passages alongside matching image regions and tables in one result. " +
      "Use it when a question is about something on a page that text alone cannot find: a tag on " +
      "a P&ID, a row in a datasheet, a symbol on a scanned drawing. " +
      "Do not use it where a plain text search would do — knowledge.search_authorized is faster and " +
      "does not pull image regions the model will then have to read. " +
      "Effects: none. It only reads, needs nobody's approval, and touches no network. " +
      "Limits: at most 4 passages, 4 image regions, and 4 tables per call. Each result carries its " +
      "own citation marker: [E#] for a text passage, [I#] for an image region (with page and " +
      "bounding box), [T#] for a table (with headers and rows preserved as structure). " +
      "If it returns nothing: nothing matched. The same clearance that filters the prose index " +
      "filters the multimodal index, so an empty result may also mean the matching material is " +
      "outside what the signed-in person is cleared to read — say no source was found rather than " +
      "guessing.",
    parameters: closed({
      query: Type.String({
        description:
          "What to look for, in natural language. Specific technical terms retrieve better than paraphrase.",
        minLength: 1,
      }),
      documentType: Type.Optional(
        Type.Union(
          [
            Type.Literal("pid"),
            Type.Literal("datasheet"),
            Type.Literal("sop"),
            Type.Literal("vendor_quote"),
            Type.Literal("report"),
          ],
          {
            description:
              "Optional: narrow to one document type. Use this for P&ID-specific queries " +
              "(instrument tag, valve, line number) and for datasheet queries that are " +
              "looking for a table cell, not prose.",
          },
        ),
      ),
      documentSha256: Type.Optional(
        Type.String({
          description: "Optional: limit the search to a single document the asker has already cited.",
          minLength: 1,
        }),
      ),
      maxResults: Type.Optional(
        Type.Integer({
          minimum: 1,
          maximum: 6,
          description:
            "How many results of each kind (text, image, table) to return. Defaults to 4.",
        }),
      ),
    }),
  },

  {
    name: "memory.recall_authorized",
    label: "Read remembered notes",
    readOnly: true,
    description:
      'Reads what this deployment remembers for one scope: "run" (this task\'s own state), ' +
      '"workspace" (terminology, templates and stable facts agreed for this project), or "user" ' +
      "(the signed-in person's preferences). " +
      "Use it before settling on wording a project has already agreed, so the same term does not " +
      "get re-derived differently on every task. " +
      "Do not use what it returns as a citable source: these are the deployment's own notes, " +
      "not retrieved passages, and a claim that needs a citation still needs a search. " +
      "Effects: none. It only reads, needs nobody's approval, and touches no network. " +
      "Limits: you cannot name a project or a person — both are taken from who is signed in, and " +
      "you are shown only what they are cleared to read. " +
      "If it returns nothing: nothing has been agreed for that scope. Proceed and say what you " +
      "assumed.",
    parameters: closed({
      scope: Type.Union(
        [Type.Literal("run"), Type.Literal("workspace"), Type.Literal("user")],
        { description: "Which scope to read." },
      ),
    }),
  },

  {
    name: "capability.search",
    label: "Find guidance",
    readOnly: true,
    description:
      "Lists the skills installed on this machine that match a description, as a name, a summary " +
      "and a version. " +
      "Use it when a task looks like one somebody has written guidance for — a template to " +
      "follow, a checklist, a house convention — before inventing your own approach. " +
      "Do not use it to read instructions: it returns descriptions only, and reading a skill's " +
      "instructions is a separate, deliberate step. " +
      "Effects: none. It only reads local metadata, needs nobody's approval, and touches no " +
      "network. " +
      "Limits: filtered to what the signed-in person may see and to what this task is permitted " +
      "to do, so a skill needing a tool you do not have is not offered. " +
      "If it matches nothing: there is no installed guidance for this. Carry on with the task.",
    parameters: closed({
      query: Type.String({
        description: "What kind of guidance you are looking for, in a few words.",
        minLength: 1,
      }),
    }),
  },

  {
    name: "sovereignty.get_evidence",
    label: "Read the machine's network record",
    readOnly: true,
    description:
      "Returns this machine's own record of every outbound connection attempted since it started, " +
      "and which of them were refused, together with the operating mode it is in. " +
      "Use it when somebody asks whether anything left this machine, or asks you to substantiate " +
      "that the work stayed local. " +
      "Do not use it to decide whether you may do something — that is the gateway's job, and this " +
      "only reports what already happened. " +
      "Effects: none. It reads a record this machine keeps about itself. " +
      "Limits: it lists the most recent 20 attempts but always counts all of them, so the number " +
      "is exact even when the list is not complete. " +
      "If it reports nothing: no outbound call has been attempted. That is the whole record, not " +
      "a summary of it.",
    parameters: closed({}),
  },

  {
    name: "workspace.read_text",
    label: "Read a file",
    readOnly: true,
    description:
      "Reads a text file from this task's own working directory. " +
      "Use it to re-read a draft you wrote earlier in this task. " +
      "Do not use it for PDFs, images or anything you did not put there: only this task's " +
      "directory is readable, any other path is refused, and documents are read with the " +
      "knowledge tools instead. " +
      "Effects: none. It only reads, needs nobody's approval, and touches no network. " +
      'Limits: give a relative name such as "draft.md". Without fromLine you get the beginning of ' +
      "the file and a note saying how much was left; with fromLine you get a named window of at " +
      "most 400 lines. " +
      "If it says the file does not exist: you have not written it in this task. Do not describe " +
      "its contents.",
    parameters: closed({
      path: Type.String({
        description: 'Relative to the task\'s working directory, for example "draft.md".',
        minLength: 1,
      }),
      fromLine: Type.Optional(
        Type.Integer({
          minimum: 1,
          description:
            "First line to read, 1-based. Supply it to read a named window instead of the start of the file.",
        }),
      ),
      maxLines: Type.Optional(
        Type.Integer({
          minimum: 1,
          maximum: 400,
          description: "How many lines to read from fromLine. Defaults to 400.",
        }),
      ),
    }),
  },

  {
    name: "calculation.evaluate_with_units",
    label: "Calculate",
    readOnly: true,
    description:
      "Evaluates an arithmetic expression with units, deterministically, and returns the result " +
      "with every step of the working. " +
      "Use it for any number that will appear in a deliverable. A figure you worked out in your " +
      "head is not verifiable and may be wrong; one from here is recorded and can be shown. " +
      "Do not use your own arithmetic to recompute or re-round what it returns — quote the result exactly as given. " +
      "Effects: none that a person must approve, but each call is recorded and the workbook tool " +
      "draws on that record, so the order you run them in is the order they appear. " +
      "Limits: arithmetic with units, not algebra or code. " +
      "If it cannot parse the expression: rewrite it with explicit units and operators, for " +
      'example "1500 kg / 3 m^3" rather than "1500kg per 3 cubic metres".',
    parameters: closed({
      expression: Type.String({
        description: 'With units, for example "1500 kg / 3 m^3" or "0.85 * 240 kW".',
        minLength: 1,
      }),
    }),
  },

  {
    name: "artifact.verify_docx",
    label: "Check a produced file",
    readOnly: true,
    description:
      "Re-opens a file this task produced and reports what is actually inside it — the sections " +
      "that are really there, not the ones the write call claimed. " +
      "Use it after producing a document or workbook and before telling anybody it is ready. " +
      "Do not use it on a file this task did not create; it exists to check your own output. " +
      "Effects: none. It only reads, and needs nobody's approval. " +
      "Limits: this task's working directory only. " +
      "If it reports a section missing: say so and fix it. Do not describe the document as ready.",
    parameters: closed({
      path: Type.String({
        description: "Relative to the task's working directory.",
        minLength: 1,
      }),
    }),
  },

  {
    name: "agent.delegate_readonly",
    label: "Delegate a read-only sub-task",
    readOnly: true,
    description:
      "Hands one bounded, read-only piece of work to a narrower worker and returns its findings. " +
      "Use it when a sub-task is separable and would otherwise fill your own context — checking a " +
      "set of figures, or retrieving across several documents at once. " +
      "Do not use it for anything that writes, and do not use it to escape a limit: a child is " +
      "never more capable than you are, and is given strictly fewer tools. " +
      "Effects: none outside this task. The child cannot write, produce a document or run code, " +
      "which is why it needs nobody's approval. " +
      "Limits: one level deep — a child cannot delegate further — and it is bounded by its own " +
      "time and step budget. " +
      "If it fails or times out: you get what it had reached. Do the work yourself or say " +
      "what could not be checked; do not present its partial findings as complete.",
    parameters: closed({
      profile: Type.Union(
        [
          Type.Literal("knowledge-retriever"),
          Type.Literal("document-extractor"),
          Type.Literal("calculation-checker"),
          Type.Literal("artifact-reviewer"),
        ],
        { description: "Which worker to use. Each has its own narrow tool list." },
      ),
      task: Type.String({
        description:
          "The one thing this worker should establish, in a sentence. It cannot see your conversation.",
        minLength: 1,
      }),
    }),
  },

  {
    name: "memory.promote_approved",
    label: "Record an approved fact",
    readOnly: false,
    description:
      "Copies one fact this task already holds into the project's memory, where later tasks will " +
      "read it. " +
      "Use it only for a stable project fact somebody has explicitly approved recording — an " +
      "agreed term, a settled convention. " +
      "Do not use it for a figure quoted from a restricted document, or for anything you merely " +
      "judged useful: promotion publishes, quietly and permanently. " +
      "Effects: writes something later tasks read. It requires the id of an approval a person " +
      "granted for that exact value, and the value is taken from what this task recorded rather " +
      "than from anything you write here. " +
      "Limits: if the value has changed since the approval, the call is refused and a new " +
      "approval is needed. " +
      "If it is refused: do not rephrase and retry. Say that the fact was not recorded and why.",
    parameters: closed({
      key: Type.String({
        description: "The key this task recorded the fact under.",
        minLength: 1,
      }),
      approvalId: Type.String({
        description: "The id of the granted approval for this exact fact.",
        minLength: 1,
      }),
    }),
  },

  {
    name: "workspace.write_text",
    label: "Write a file",
    readOnly: false,
    description:
      "Writes a text file into this task's own working directory. " +
      "Use it for notes and drafts — working material, not the thing somebody will be handed. " +
      "Do not use it for deliverables: a document a person receives is produced with " +
      "artifact.create_approval_note, which applies the template and the DRAFT marking. " +
      "Effects: creates or overwrites a file. A person must approve it before it happens, so " +
      "expect a pause; nothing is written until they answer. " +
      'Limits: relative names inside this task\'s directory only. Any other path is refused. ' +
      "If it is declined: do not write it elsewhere or under another name. Say it was not " +
      "written and carry on with what you can do.",
    parameters: closed({
      path: Type.String({
        description: 'Relative to the task\'s working directory, for example "draft.md".',
        minLength: 1,
      }),
      content: Type.String({ description: "The complete file contents." }),
    }),
  },

  {
    name: "artifact.create_approval_note",
    label: "Produce an approval note",
    readOnly: false,
    description:
      "Produces a Word document from an installed template, with every field filled from what you " +
      "supply. " +
      "Use it for the deliverable a person will actually be handed. " +
      "Do not use it for working notes, and do not describe the document as final: the result is " +
      "marked DRAFT until somebody signs it. " +
      "Effects: creates a file. A person must approve it before it happens, so expect a pause. " +
      "Limits: every field the template asks for must be present, as text; a missing required " +
      "field fails the render rather than producing a document with a gap in it. Available " +
      "templates: approval_note. " +
      "If it fails to render: it names the field that was missing. Supply it and call again — do " +
      "not claim a document was produced. Afterwards, check it with artifact.verify_docx before " +
      "saying it is ready.",
    parameters: closed({
      path: Type.String({
        description: 'Relative, for example "approval-note.docx".',
        minLength: 1,
      }),
      template: Type.Union([Type.Literal("approval_note")], {
        description: "Which installed template to render.",
      }),
      content: Type.Record(Type.String(), Type.String(), {
        description: "Field name to text. Every value must be a string.",
      }),
    }),
  },

  {
    name: "artifact.create_calculation_workbook",
    label: "Produce a calculation workbook",
    readOnly: false,
    description:
      "Produces an Excel workbook showing the working behind every figure this task calculated, " +
      "as live formulas Excel recomputes. " +
      "Use it when somebody needs to check the arithmetic rather than take it on trust. " +
      "Do not use it to pass figures directly: it draws on the calculation.evaluate_with_units calls " +
      "already made, so run the calculations first or the workbook will be empty. " +
      "Effects: creates a file. A person must approve it before it happens, so expect a pause. " +
      "Limits: it can only show what was actually calculated through the engine. Arithmetic you " +
      "did in your head does not appear, because there is no working to show. " +
      "If it comes out empty: you have not run any calculations. Run them, then call this again.",
    parameters: closed({
      path: Type.String({ description: 'Relative, for example "working.xlsx".', minLength: 1 }),
    }),
  },

  {
    name: "artifact.create_briefing_deck",
    label: "Produce a briefing deck",
    readOnly: false,
    description:
      "Produces a PowerPoint briefing from a fixed four-section template: Findings, " +
      "Recommendation, Assumptions, Evidence, in that order. " +
      "Use it when somebody needs to present what this task found, rather than read it. " +
      "Do not use it to write a document — artifact.create_approval_note produces prose, and a " +
      "deck of full paragraphs is neither. You supply the bullets; you do not choose the " +
      "headings, and each of the four needs at least one. Evidence is required for the reason a " +
      "citation is: a briefing whose findings have no source is not a briefing. " +
      "Effects: creates a file. A person must approve it before it happens, so expect a pause. " +
      "Limits: the deck is marked DRAFT until somebody signs it, and the word is printed on the " +
      "slide rather than only stored. " +
      "If it refuses a section as empty: that section had nothing under it. A heading with no " +
      "bullets reads as a finding of nothing, which is not the same as having nothing to say.",
    parameters: closed({
      path: Type.String({ description: 'Relative, for example "briefing.pptx".', minLength: 1 }),
      content: Type.Object(
        {
          title: Type.String({ description: "Deck title, shown on the cover slide.", minLength: 1 }),
          findings: Type.Array(Type.String(), { description: "What this task established." }),
          recommendation: Type.Array(Type.String(), { description: "What should happen next." }),
          assumptions: Type.Array(Type.String(), { description: "What was taken as given." }),
          evidence: Type.Array(Type.String(), { description: "Where the findings came from." }),
        },
        {
          description:
            "Title plus one list of bullet strings per section. A single string is accepted " +
            "where a section is one sentence.",
        },
      ),
    }),
  },

  {
    name: "sandbox.run_code",
    label: "Run code in the sandbox",
    readOnly: false,
    description:
      "Runs a short program in a container with the network switched off, a read-only root " +
      "filesystem, capped CPU and memory, no host credentials, and a wall-clock ceiling. " +
      "Use it only when a result genuinely needs code. " +
      "Do not use it for arithmetic — calculation.evaluate_with_units does that deterministically " +
      "and records the working. " +
      "Effects: executes code. A person must approve it before it happens. " +
      "Limits: it runs only where a container runtime is responding and the base image is already " +
      "on the machine. ARJUN never fetches an image, because that would be an outbound call. On a " +
      "machine with weaker isolation than a container, the call is refused rather than run. " +
      "If it is refused: treat the refusal as final. Say the code was not run — never describe " +
      "what it would have produced, and never present imagined output as a result.",
    parameters: closed({
      language: Type.Union([Type.Literal("python"), Type.Literal("javascript")], {
        description: "Which language runtime to use.",
      }),
      source: Type.String({ description: "The complete program.", minLength: 1 }),
    }),
  },
] as const;

/** The definition for one wire name, if this runtime knows it. */
export function definitionFor(name: string): ToolDefinition | undefined {
  return TOOL_DEFINITIONS.find((definition) => definition.name === name);
}
