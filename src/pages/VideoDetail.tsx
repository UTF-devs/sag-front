import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  Eye,
  Clock,
  Send,
  Link as LinkIcon,
  ArrowLeft,
} from "lucide-react";
import { ContactInfoSection } from "../screens/HomePage/sections/ContactInfoSection";
import { useLanguage } from "../contexts/LanguageContext";
import { client } from "../services";

interface LinkItem {
  id: number;
  link: string;
  icon: {
    name: string;
    image: string;
  };
}

interface VideoData {
  id: number;
  title: string;
  description: string;
  content: string;
  created_at: string;
  links: LinkItem[];
  view_count: number;
}

const VideoDetail = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { t, language } = useLanguage();

  const [video, setVideo] = useState<VideoData | null>(null);

  const mapLang = (lang: string) =>
    lang === "rus" ? "ru" : lang === "uzb" ? "uz" : "en";

  useEffect(() => {
    const fetchVideo = async () => {
      try {
        const lang = mapLang(language);
        const res = await client.get(`/${lang}/api/v1/blog/get_blog_by_id/${id}/`);
        setVideo(res.data);
      } catch (err) {
        console.error("Video fetch error:", err);
      }
    };

    if (id) {
      fetchVideo();
    }
  }, [id, language]);

  if (!video) return null;

  return (
    <div className="md:pt-28 pt-24 bg-[#FFFCE0] min-h-screen flex flex-col">
      <Navbar />

      <div className="container mx-auto px-4 py-8 flex-1">
        {/* Back Button - Mobile Only */}
        <button
          onClick={() => navigate("/videos")}
          className="flex items-center gap-2 text-gray-600 hover:text-gray-800 mb-4 transition-colors"
        >
          <ArrowLeft size={20} />
          <span className="md:text-2xl">{t("video.back")}</span>
        </button>

        <div className="flex items-center justify-between mb-6">
          <h1 className="md:text-3xl text-xl font-semibold text-gray-800">{video.title}</h1>
        </div>

        <div className="mb-4">
          <video
            src={video.content}
            controls
            className="w-full h-[300px] md:h-[600px] object-cover rounded-lg"
            onError={(e) => console.error("Video load error:", e)}
            playsInline
          >
            Your browser does not support the video tag.
          </video>
        </div>

        <div className="flex flex-wrap items-center justify-between text-[#8A6E4F] text-sm mb-6">
          <div className="flex items-center space-x-4">
            <div className="flex items-center gap-1">
              <Clock size={16} />
              <span>{video.created_at?.slice(0, 10)}</span>
            </div>
            <div className="flex items-center gap-1">
              <Eye size={16} />
              <span>{video.view_count}</span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {Array.isArray(video.links) &&
              video.links.map((item) => (
                <a
                  key={item.id}
                  href={item.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:opacity-75 transition"
                  title={item.icon?.name}
                >
                  <img
                    src={item.icon?.image}
                    alt={item.icon?.name || "icon"}
                    className="w-5 h-5 object-contain"
                  />
                </a>
              ))}
          </div>
        </div>

        <p className="text-gray-700 text-base">
          {/* Optional description or content about the video */}
          {video.description}
        </p>
      </div>

      <ContactInfoSection />
    </div>
  );
};

export default VideoDetail;
