import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ContactInfoSection } from "../screens/HomePage/sections/ContactInfoSection";
import { useLanguage } from "../contexts/LanguageContext";
import { client } from "../services";

interface BlogVideo {
  id: number;
  title: string;
  content: string; // video URL
}

const Videos = () => {
  const { t, language } = useLanguage();
  const [videoData, setVideoData] = useState<BlogVideo[]>([]);

  const mapLang = (lang: string) =>
    lang === "rus" ? "ru" : lang === "uzb" ? "uz" : "en";

  useEffect(() => {
    const fetchVideos = async () => {
      try {
        const lang = mapLang(language);
        const res = await client.get(`/${lang}/api/v1/blog/get_all_blogs/`);
        setVideoData(res.data);
      } catch (error) {
        console.error("Error fetching videos:", error);
      }
    };

    fetchVideos();
  }, [language]);

  return (
    <div className="md:pt-28 pt-24 min-h-screen bg-[#FFFCE0]">
      <div className="container mx-auto px-4 py-8">
        <h1 className="md:text-3xl text-lg font-semibold text-gray-800 mb-6 text-center">
          {t("video.title")}
        </h1>
        <div className="grid sm:grid-cols-2 md:grid-cols-4  min-h-[500px] gap-4">
          {videoData.map((video) => (
            <Link to={`/videos/${video.id}`} key={video.id} className="block">
              <div className="relative overflow-hidden">
                <video
                  src={video.content}
                  className="w-full h-[271px] object-cover"
                  muted
                  preload="metadata"
                  playsInline
                />
                <div className="absolute top-3 left-2 text-white p-1 rounded text-sm line-clamp-3 md:text-lg bg-black bg-opacity-50">
                  {video.title}
                </div>
              </div>
            </Link>
          ))}
        </div>
      </div>
      <ContactInfoSection />
    </div>
  );
};

export default Videos;
