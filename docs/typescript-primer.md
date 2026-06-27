# TypeScript primer (for this codebase)

A practical introduction to the TypeScript you'll meet in `frontend/`, taught
through the patterns this repo actually uses. It assumes you know Python and
some JavaScript. Every snippet is real (or lightly trimmed) code from the app.

The one idea to anchor everything: **TypeScript is JavaScript plus a
compile-time type checker.** The types help your editor and catch mistakes
*before* you run; then the compiler **erases all of them** and ships plain
JavaScript. Nothing about types exists at runtime. Most of the "defensive"
code in this repo follows from that single fact (see §8).

---

## 1. Type annotations

You attach a type to a variable, parameter, or return value with `: Type`.

```ts
const name: string = "fishhunter";
function double(n: number): number { return n * 2; }
```

You rarely need to annotate local variables — TS *infers* them
(`const n = 2` is already `number`). Annotate the **boundaries**: function
parameters and return types, and data crossing in/out of the app.

Primitive types: `string`, `number`, `boolean`, `null`, `undefined`, plus
`void` (returns nothing), `unknown` (anything, but you must check before use),
and `any` (anything, checks off — avoid it).

---

## 2. `interface` — the shape of an object

An `interface` names the shape of an object. It's the TS equivalent of "this
dict must have these keys with these types." From `api/agentStream.ts`:

```ts
export interface AgentChatPayload {
  message: string;
  history: { role: string; content: string }[];
  model?: string | null;
  dashboard_config?: string | null;
  dashboard_state: Record<string, unknown>;
}
```

- `?` makes a field **optional** — `model?: ...` may be omitted entirely.
- `{ role: string; content: string }[]` — the trailing `[]` means "array of";
  `history` is a list of those objects.
- Interfaces are **erased at runtime** — purely a compile-time contract.

`type` aliases do a similar job (`type Foo = { ... }`); this codebase uses
`interface` for object shapes and `type` for unions and aliases (next).

---

## 3. Union types and literal types

`A | B` means "one of these." Hugely common.

```ts
model?: string | null;          // a string, or null
```

**Literal types** restrict to specific values — like a Python `Literal`:

```ts
export type ReportLanguage = "en" | "zh-Hans" | "zh-Hant";
type Granularity = "day" | "week" | "month";
```

A value of type `ReportLanguage` can *only* be one of those three strings; any
other string is a compile error.

---

## 4. Generics — `Type<Param>`

The `<...>` is a **type parameter** — a type that's filled in per use, like a
typed container. You read it as "X of Y."

```ts
Promise<void>                 // a promise that resolves to nothing
Record<string, unknown>       // an object: string keys, unknown values
GroupStat[]                   // (shorthand for Array<GroupStat>)
Partial<ReportSpec>           // ReportSpec with every field made optional
```

`Record<K, V>` and `Partial<T>` are **built-in utility types**. `Partial` shows
up in our store update methods — "patch some fields of a ReportSpec":

```ts
patchSpec: (patch: Partial<ReportSpec>) => void;
```

---

## 5. Function types (the callback pattern)

A field (or parameter) can itself be a function. The type is
`(args) => ReturnType`. From `AgentStreamHandlers`:

```ts
export interface AgentStreamHandlers {
  onToken: (text: string) => void;
  onClarify: (question: string, options: string[]) => void;
  onDone: (elapsedMs: number) => void;
}
```

`onToken` is "a function taking a `string`, returning `void`." The caller
supplies these callbacks and `streamAgentChat` invokes the right one per event
— how you pass *behavior* into a function.

---

## 6. `async` / `await` and `Promise<T>`

Same model as Python's `async`/`await`. An `async` function always returns a
`Promise<T>`; `await` unwraps it.

```ts
export async function streamAgentChat(
  payload: AgentChatPayload,
  on: AgentStreamHandlers,
  signal: AbortSignal,
): Promise<void> {
  const resp = await fetch("/api/agent/chat", { method: "POST", /* ... */ });
  // ...
}
```

`Promise<void>` = "resolves with no value." `AbortSignal` is a browser type for
cancellation (our stop button / component unmount aborts the fetch).

---

## 7. Discriminated unions + narrowing (the FigureSource pattern)

This is the most important *design* pattern in the app. Several object shapes
share a common literal field (the **discriminant**, here `kind`):

```ts
export interface DateFigureSource    { kind: "stats-by-date";   /* ... */ }
export interface GroupFigureSource   { kind: "stats-by-group";  /* ... */ }
export interface DeepdiveFigureSource { kind: "stats-deepdive"; /* ... */ }

export type FigureSource = DateFigureSource | GroupFigureSource | DeepdiveFigureSource;
```

When you check `kind`, TS **narrows** the type — inside the `if`, it knows
exactly which shape you have and lets you access that shape's fields safely
(`reportStore.ts`):

```ts
function effectiveSource(fig: ReportFigure, period: DateRange | null | undefined): FigureSource {
  const src = fig.source;                       // src: FigureSource (could be any of 3)
  if (!fig.inherit_period || !period?.start || !period?.end) return src;
  if (src.kind === "stats-by-date")             // narrowed: src is DateFigureSource here
    return { ...src, date_from: period.start, date_to: period.end };
  return { ...src, ranges: [{ start: period.start, end: period.end }] }; // the ranged kinds
}
```

This is how the Report tab handles three different chart recipes with one type.
(`?.` is **optional chaining** — `period?.start` is `undefined` if `period` is
null/undefined instead of throwing.)

---

## 8. Types are erased — so validate at the boundary

Because types vanish at runtime, data arriving over the network is **not**
guaranteed to match its TypeScript type — TS can't enforce what the server
sends. So at every external boundary we coerce/validate with real JavaScript.
From the SSE dispatch in `agentStream.ts`:

```ts
parsed = data ? (JSON.parse(data) as Record<string, unknown>) : {};
on.onToken(String(parsed.text ?? ""));
on.onClarify(String(parsed.question ?? ""),
             Array.isArray(parsed.options) ? parsed.options.map(String) : []);
```

- `as Record<string, unknown>` is a **type assertion** — "compiler, trust me,
  treat this as this type." It does **no runtime check**; use sparingly.
- `?? ""` is **nullish coalescing** — use the right side if the left is
  `null`/`undefined` (not just falsy, unlike `||`).
- `String(...)`, `Number(...)`, `Array.isArray(...)` are genuine runtime
  guards. **Typed inside, validated at the edges** is the idiom.

You'll also see `as unknown as X` (a "double assertion") at the API client when
a generated wire type doesn't line up with the hand-written app type:

```ts
return data as unknown as ReportSpec;   // client.ts
```

It means "I know these don't structurally match; force it." A code smell in
general, but deliberate here (the OpenAPI generator marks fields optional that
the server always fills — see `api/types.ts` comments).

---

## 9. The generated API client (openapi-fetch)

The backend's OpenAPI schema is compiled into `frontend/src/api/schema.d.ts`
(via `make openapi`), and the client is created against it:

```ts
import createClient from "openapi-fetch";
import type { components, paths } from "./schema";

const client = createClient<paths>({ baseUrl: BASE });
```

`createClient<paths>` means every request path, its params, body, and response
are type-checked against the backend contract. If the backend changes a route
and you regenerate, the frontend stops compiling where it's now wrong — that's
the safety net (and the CI "openapi-drift" gate keeps the schema in sync).

Convenience aliases re-export generated types under friendly names
(`api/types.ts`):

```ts
type Schemas = components["schemas"];
export type GroupStat = Schemas["GroupStat"];
export type DeepdivePanel = DeepdiveRequest["panel"];  // index into another type
```

`Type["field"]` is an **indexed access type** — "the type of that field." So
`DeepdiveRequest["panel"]` is whatever literal-union the backend declared for
`panel`, without restating it.

---

## 10. Zustand store typing (how app state is typed)

The app's state lives in Zustand stores. You define a `State` interface (data +
the methods that mutate it), then create the store with that type:

```ts
interface ReportState {
  spec: ReportSpec;
  figureData: Record<string, FigureData>;
  patchSpec: (patch: Partial<ReportSpec>) => void;
  addFigure: (source: FigureSource, title: string) => void;
  renderFigure: (id: string) => Promise<void>;
  // ...
}

export const useReportStore = create<ReportState>((set, get) => ({
  spec: emptySpec(),
  figureData: {},
  patchSpec: (patch) => set((s) => ({ spec: { ...s.spec, ...patch } })),
  // ...
}));
```

- `create<ReportState>(...)` ties the whole store to the interface — every
  field and method is checked against it.
- `set`/`get` update and read state. `set((s) => ({ ... }))` returns a
  *partial* new state that's merged in.
- In components, `useDashboardStore((s) => s.configId)` selects one slice; the
  selector's return type is inferred, so `configId` keeps its type downstream.

---

## Cheat sheet

| Syntax | Meaning | Python analogy |
|---|---|---|
| `x: string` | type annotation | `x: str` |
| `field?: T` | optional field | `field: T \| None` (roughly) |
| `A \| B` | union | `Union[A, B]` |
| `"a" \| "b"` | literal union | `Literal["a", "b"]` |
| `T[]` / `Array<T>` | array | `list[T]` |
| `Record<K, V>` | keyed object | `dict[K, V]` |
| `Partial<T>` | all fields optional | — |
| `Promise<T>` | async result | `Awaitable[T]` |
| `(a: T) => R` | function type | `Callable[[T], R]` |
| `x as T` | type assertion (no runtime check) | `cast(T, x)` |
| `a ?? b` | b if a is null/undefined | `a if a is not None else b` |
| `a?.b` | safe access | `a.b if a is not None else None` |
| `...obj` | spread (copy fields) | `{**obj}` |

## Where to look next in this repo
- `api/agentStream.ts` — interfaces, callbacks, async, boundary validation (§2, 5, 6, 8).
- `api/types.ts` — discriminated unions, generated-type aliases (§7, 9).
- `store/reportStore.ts` / `store/dashboardStore.ts` — store typing (§10).
- `charts/*Option.ts` — typed chart builders (`EChartsOption`), and the
  occasional `any` escape hatch where a library's types are too strict.
