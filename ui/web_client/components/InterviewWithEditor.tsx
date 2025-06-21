import React, { useState, Children, cloneElement, isValidElement } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import SimpleCodeEditor from "./SimpleCodeEditor";

export type InterviewWithEditorProps = {
  children: React.ReactNode;
  initialCode?: string;
  onSubmit: (code: string) => void;
  codeLanguage?: string;
  highlight?: boolean;
};

const InterviewWithEditor: React.FC<InterviewWithEditorProps> = ({
  children,
  initialCode = "",
  onSubmit,
  codeLanguage = "text",
  highlight = true,
}) => {
  const [editorOpen, setEditorOpen] = useState(false);
  const [code, setCode] = useState(initialCode);

  // Clone children to inject editorOpen prop
  const childrenWithProps = Children.map(children, (child) => {
    if (!isValidElement<{ editorOpen?: boolean }>(child)) return child;
    return cloneElement(child, { editorOpen });
  });

  if (!editorOpen) {
    return (
      <div className="relative w-full h-full flex flex-col">
        <div className="flex-1 min-w-0">{childrenWithProps}</div>
        <div className="p-4 flex justify-center">
          <button
            className="px-4 py-2 rounded bg-white text-black font-semibold shadow"
            onClick={() => setEditorOpen(true)}
          >
            Open Code Editor
          </button>
        </div>
      </div>
    );
  }

  return (
    <PanelGroup direction="horizontal" className="flex-grow flex w-full h-full">
      <Panel defaultSize={30} minSize={20}>
        <div className="bg-black flex flex-col h-full min-w-0" style={{ overflow: "auto" }}>
          <div className="flex-1 min-w-0">{childrenWithProps}</div>
        </div>
      </Panel>
      <PanelResizeHandle className="w-1 bg-gray-600 cursor-col-resize" />
      <Panel defaultSize={70} minSize={20}>
        <motion.div
          key="editor"
          initial={{ x: 100, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          exit={{ x: 100, opacity: 0 }}
          transition={{ type: "spring", stiffness: 200, damping: 30 }}
          className="bg-[#181818] flex flex-col h-full min-w-0"
          style={{ boxShadow: "0 0 24px 0 rgba(0,0,0,0.2)" }}
        >
          <div className="flex items-center justify-between p-4 border-b border-gray-700">
            <span className="font-semibold text-white">Code Editor</span>
            <button
              className="px-3 py-1 rounded bg-gray-800 text-white hover:bg-gray-700"
              onClick={() => setEditorOpen(false)}
            >
              Back to Interview
            </button>
          </div>
          <div className="flex-1 overflow-auto p-2">
            <SimpleCodeEditor
              value={code}
              onChange={setCode}
              language={codeLanguage}
              highlight={highlight}
              minRows={16}
              style={{
                background: "#23272a",
                color: "#fff"
              }}
            />
          </div>
          <div className="p-4 border-t border-gray-700 flex justify-end">
            <button
              className="px-4 py-2 rounded bg-blue-600 text-white font-semibold shadow"
              onClick={() => onSubmit(code)}
            >
              Submit
            </button>
          </div>
        </motion.div>
      </Panel>
    </PanelGroup>
  );
};

export default InterviewWithEditor;
