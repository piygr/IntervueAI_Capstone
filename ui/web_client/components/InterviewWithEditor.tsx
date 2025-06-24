import React, { useState, Children, cloneElement, isValidElement } from "react";
import { motion } from "framer-motion";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import SimpleCodeEditor from "./SimpleCodeEditor";

export type InterviewWithEditorProps = {
  children: React.ReactNode;
  code: string;
  onCodeChange: (code: string) => void;
  onSubmit: (code: string) => void;
  codeLanguage?: string;
  highlight?: boolean;
  isSubmitting?: boolean;
};

const InterviewWithEditor: React.FC<InterviewWithEditorProps> = ({
  children,
  code,
  onCodeChange,
  onSubmit,
  codeLanguage = "text",
  highlight = true,
  isSubmitting = false
}) => {
  const [editorOpen, setEditorOpen] = useState(false);

  const childrenWithProps = Children.map(children, (child) => {
    if (!isValidElement<{ editorOpen?: boolean }>(child)) return child;
    return cloneElement(child, { editorOpen });
  });

  const buttonBase = "transition duration-150 focus:outline-none font-semibold rounded shadow";

  if (!editorOpen) {
    return (
      <div className="relative w-full h-full flex flex-col">
        <div className="flex-1 min-w-0">{childrenWithProps}</div>
        <div className="p-4 flex justify-center">
          <button
            className={`${buttonBase} bg-white text-black px-4 py-2 hover:bg-gray-100 active:bg-gray-200 focus:ring-2 focus:ring-gray-400`}
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
        <div className="bg-black flex flex-col h-full min-w-0 overflow-auto">
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
              className={`${buttonBase} bg-gray-800 text-white px-3 py-1 hover:bg-gray-700 active:bg-gray-900 focus:ring-2 focus:ring-gray-400`}
              onClick={() => setEditorOpen(false)}
            >
              Close Editor
            </button>
          </div>
          <div className="flex-1 overflow-auto p-2">
            <SimpleCodeEditor
              value={code}
              onChange={onCodeChange}
              language={codeLanguage}
              highlight={highlight}
              minRows={16}
              style={{ background: "#23272a", color: "#fff" }}
            />
          </div>
          <div className="p-4 border-t border-gray-700 flex justify-end">
            <button
              disabled={isSubmitting}
              className={`${buttonBase} ${isSubmitting ? 'opacity-50 cursor-not-allowed' : ''} bg-indigo-600 text-white px-5 py-2 rounded-md text-sm font-semibold hover:bg-indigo-700 transition-colors`}
              onClick={() => {
                if (window.confirm("Are you sure you want to submit this code?")) {
                  onSubmit(code);
                }
              }}
            >
              {isSubmitting ? 'Submitting...' : 'Submit'}
            </button>
          </div>
        </motion.div>
      </Panel>
    </PanelGroup>
  );
};

export default InterviewWithEditor;
