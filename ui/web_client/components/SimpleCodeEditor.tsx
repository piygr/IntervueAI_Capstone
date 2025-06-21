import React, { useRef } from "react";
import dynamic from "next/dynamic";
import type * as CodeEditorModule from "@uiw/react-textarea-code-editor";

type CodeEditorType = typeof CodeEditorModule.default;

const CodeEditor = dynamic<CodeEditorType>(
  () => import("@uiw/react-textarea-code-editor").then(mod => mod.default),
  { ssr: false, loading: () => <textarea rows={10} style={{ width: "100%" }} /> }
);

type SimpleCodeEditorProps = {
  value: string;
  onChange: (val: string) => void;
  language?: string;
  highlight?: boolean;
  style?: React.CSSProperties;
  className?: string;
  minRows?: number;
};

const SimpleCodeEditor: React.FC<SimpleCodeEditorProps> = ({
  value,
  onChange,
  language = "text",
  highlight = true,
  style,
  className,
  minRows = 10,
}) => {
  const editorRef = useRef<HTMLTextAreaElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Prevent copy/paste from outside the editor
  const handlePaste = (e: React.ClipboardEvent<HTMLTextAreaElement>) => {
    // Only allow paste if the focus is already inside the editor
    if (document.activeElement !== editorRef.current) {
      alert("Pasting is not allowed")
      e.preventDefault();
    }
    // Only allow paste if the clipboard data comes from the editor itself
    // (This is a best-effort, as browsers don't provide a perfect way to check this)
    // You may want to show a message here if needed
  };

  const handleCopy = (e: React.ClipboardEvent<HTMLTextAreaElement>) => {
    // Only allow copy if the focus is inside the editor
    if (document.activeElement !== editorRef.current) {
      alert("Copying/Cutting is not allowed")
      e.preventDefault();
    }
  };

  if (highlight) {
    return (
    <div ref={containerRef} className="h-full overflow-auto">
        <CodeEditor
            value={value}
            language={language}
            placeholder="Type your code here…"
            onChange={(ev: React.ChangeEvent<HTMLTextAreaElement>) => onChange(ev.target.value)}
            onInput={() => {
                requestAnimationFrame(() => {
                  const el = containerRef.current;
                  if (!el) return;
              
                  const isNearBottom =
                    el.scrollTop + el.clientHeight >= el.scrollHeight - 25;
              
                  if (isNearBottom) {
                    el.scrollTop = el.scrollHeight;
                  }
                });
              }}
            padding={12}
            // minHeight={minRows * 20}
            style={{
            width: "100%",
            minHeight: "100%",
            fontFamily: "monospace",
            fontSize: 16,
            background: "#f5f5f5",
            borderRadius: 1,
            ...style,
            }}
            className={className}
            onPaste={handlePaste}
            onCopy={handleCopy}
            onCut={handleCopy}
        />
    </div>
    );
  }

  // Fallback: plain textarea
  return (
    <div ref={containerRef} className="h-full overflow-auto">
        <textarea
            ref={editorRef}
            value={value}
            onChange={e => onChange(e.target.value)}
            onInput={() => {
                if (containerRef.current) {
                containerRef.current.scrollTop = containerRef.current.scrollHeight;
                }
            }}
            rows={minRows}
            style={{
                width: "100%",
                minHeight: "100%",
                fontFamily: "monospace",
                fontSize: 16,
                background: "#f5f5f5",
                borderRadius: 6,
                padding: 12,
                ...style,
            }}
            className={className}
            onPaste={handlePaste}
            onCopy={handleCopy}
            onCut={handleCopy}
            placeholder="Type your code here…"
        />
    </div>
  );
};

export default SimpleCodeEditor;