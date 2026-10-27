import type { Readable } from 'node:stream';
import type { Dispatcher } from 'undici';
import type { HeaderMap } from '../proxy/headers.js';
import type { MessagesBody } from '../routing/messages-body.js';

/** An inbound Anthropic-format request, in both raw and parsed form. */
export interface ProviderRequest {
  readonly method: Dispatcher.HttpMethod;
  /** Path + query exactly as received (e.g. `/v1/messages?beta=true`). */
  readonly url: string;
  readonly headers: HeaderMap;
  readonly rawBody: Buffer | undefined;
  readonly body: MessagesBody | undefined;
  readonly signal: AbortSignal;
  /** Called if a response fails after streaming started (too late to fail over). */
  readonly onStreamFailure?: () => void;
  /** Timeout for the headers of this specific request. */
  readonly headersTimeoutMs?: number;
  /** Optional model override for CLI runners (e.g. Argon low/medium/high). */
  readonly modelOverride?: string | undefined;
}

/**
 * Every provider answers in the Anthropic wire format — that is the only
 * format the client understands. `unavailable` means nothing was sent to the
 * client, so the caller may safely fail over to another provider.
 */
export type ProviderResult =
  | { readonly kind: 'response'; readonly status: number; readonly headers: HeaderMap; readonly body: Readable }
  | { readonly kind: 'unavailable'; readonly reason: string };

export interface Provider {
  readonly name: string;
  send(req: ProviderRequest): Promise<ProviderResult>;
}
