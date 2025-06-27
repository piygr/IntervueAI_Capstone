"use client";
import { useEffect, useState } from "react";
import { useSearchParams, useParams } from "next/navigation";
import { CircularProgress } from "@mui/material"; // or your preferred spinner
import { motion } from "framer-motion";

type Feedback = {
  interview_summary: {
    summary_text: string;
    overall_score: number;
  };
  per_question_analysis: {
    question_number: number;
    question_text: string;
    analysis: {
      strengths: string;
      weaknesses: string;
      missing: string;
    };
    ratings: {
      clarity_of_response: number;
      technical_depth: number;
      communication_skills: number;
      relevance_to_question: number;
      overall_rating: number;
    };
  }[];
};

export default function FeedbackPage() {
  const searchParams = useSearchParams();
  const interviewId = searchParams.get('i') || '';
  const [feedback, setFeedback] = useState<Feedback | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchFeedback() {
      setLoading(true);
      try {
        const res = await fetch(`${process.env.API_HOST}/feedback/${interviewId}`);
        if (!res.ok) throw new Error("Failed to fetch feedback");
        const data = await res.json();
        setFeedback(data);
      } catch (e) {
        setFeedback(null);
      } finally {
        setLoading(false);
      }
    }
    fetchFeedback();
  }, [interviewId]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-screen bg-gradient-to-br from-blue-100 to-purple-200">
        <CircularProgress color="secondary" />
        <p className="mt-4 text-lg text-gray-700">Generating your feedback...</p>
      </div>
    );
  }

  if (!feedback) {
    return (
      <div className="flex flex-col items-center justify-center h-screen">
        <p className="text-red-500">Failed to load feedback. Please try again later.</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-100 p-8">
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-3xl mx-auto bg-white rounded-3xl shadow-2xl p-8"
      >
        <h1 className="text-4xl font-bold text-center text-purple-700 mb-6">Interview Feedback</h1>
        <div className="mb-8">
          <h2 className="text-2xl font-semibold text-gray-800 mb-2">Overall Summary</h2>
          <p className="text-lg text-gray-700 mb-2">{feedback.interview_summary.summary_text}</p>
          <div className="flex items-center gap-2">
            <span className="font-semibold text-gray-600">Overall Score:</span>
            <span className="text-2xl font-bold text-purple-600">{feedback.interview_summary.overall_score.toFixed(1)} / 5</span>
          </div>
        </div>
        <div>
          <h2 className="text-2xl font-semibold text-gray-800 mb-4">Per-Question Analysis</h2>
          <div className="space-y-6">
            {feedback.per_question_analysis.map((q, idx) => (
              <motion.div
                key={q.question_number}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.1 }}
                className="bg-gradient-to-r from-purple-100 to-blue-50 rounded-xl p-6 shadow"
              >
                <h3 className="text-lg font-bold text-purple-700 mb-2">
                  Q{q.question_number}: {q.question_text}
                </h3>
                <div className="mb-2">
                  <span className="font-semibold text-green-700">Strengths:</span>
                  <span className="ml-2 text-gray-700">{q.analysis.strengths}</span>
                </div>
                <div className="mb-2">
                  <span className="font-semibold text-red-700">Weaknesses:</span>
                  <span className="ml-2 text-gray-700">{q.analysis.weaknesses}</span>
                </div>
                <div className="mb-2">
                  <span className="font-semibold text-yellow-700">Missing:</span>
                  <span className="ml-2 text-gray-700">{q.analysis.missing}</span>
                </div>
                <div className="flex flex-wrap gap-4 mt-4">
                  <Rating label="Clarity" value={q.ratings.clarity_of_response} />
                  <Rating label="Technical Depth" value={q.ratings.technical_depth} />
                  <Rating label="Communication" value={q.ratings.communication_skills} />
                  <Rating label="Relevance" value={q.ratings.relevance_to_question} />
                  <Rating label="Overall" value={q.ratings.overall_rating} />
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </motion.div>
    </div>
  );
}

function Rating({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex flex-col items-center">
      <span className="text-xs text-gray-500">{label}</span>
      <span className="text-lg font-bold text-blue-700">{value.toFixed(1)}</span>
    </div>
  );
}