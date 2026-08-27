import type {
  AttentionArchitecture,
  AttentionRun,
} from "../types/attention";

interface ErrorResponse {
  detail?: string;
}

async function parseResponse(response: Response): Promise<AttentionRun> {
  if (!response.ok) {
    let message = "Failed to run attention.";
    try {
      const payload = (await response.json()) as ErrorResponse;
      message = payload.detail || message;
    } catch {
      // Preserve the stable fallback for non-JSON server errors.
    }
    throw new Error(message);
  }

  return (await response.json()) as AttentionRun;
}

export async function runAttention(
  text: string,
  architecture: AttentionArchitecture,
): Promise<AttentionRun> {
  const response = await fetch("/api/run", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ text, architecture }),
  });

  return parseResponse(response);
}

export async function decodeOneToken(
  sessionId: string,
  newToken?: string,
): Promise<AttentionRun> {
  const response = await fetch("/api/decode", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      session_id: sessionId,
      new_token: newToken,
    }),
  });

  return parseResponse(response);
}
