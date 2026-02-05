import { useEffect, useState } from "react";
import ProductCard from "../screens/SalesPage/ProductCard";
import { ChevronLeft } from 'lucide-react';
import { ContactInfoSection } from "../screens/HomePage/sections/ContactInfoSection";
import { Link } from "react-router-dom";
import { useLanguage } from '../contexts/LanguageContext';
import { client } from "../services";

interface Product {
  id: number;
  name: string;
  image: string;
  collection_type: string;
}

const Sales = () => {
  const { t, language } = useLanguage();
  const [data, setData] = useState<Product[]>([]);
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 21;

  useEffect(() => {
    const lang = language === 'uzb' ? 'uz' : language === 'rus' ? 'ru' : 'en';

    const fetchData = async () => {
      try {
        const res = await client.get(`/${lang}/api/v1/discount/get_main_discounted_carpets/`);
        setData(res.data || []);
      } catch (err) {
        console.error("Ma'lumotlarni yuklashda xatolik:", err);
      }
    };

    fetchData();
  }, [language]);

  const totalPages = Math.ceil(data.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const currentRugs = data.slice(startIndex, startIndex + itemsPerPage);

  return (
    <div className="bg-[#FFFCE0] md:pt-28 pt-24">
      <div className="container mx-auto px-4 py-6">
        <div className="mb-6">
          <div className="flex items-center text-base text-gray-600 mb-4">
            <ChevronLeft size={20} className="text-gray-600" /> <Link to="/">{t('nav.home')}</Link>
            <div className="pl-3 flex items-center font-semibold"> <ChevronLeft size={20} className="text-gray-600" />{t('nav.sales')}</div>
          </div>
        </div>

        <div className="w-full min-h-[500px]">
          {data.length > 0 ? (
            <>
              <div className={`grid grid-cols-2 sm:grid-cols-2 lg:grid-cols-4 gap-4`}>
                {currentRugs.map((rug) => (
                  <ProductCard
                    key={rug.id}
                    id={rug.id}
                    name={rug.name}
                    image={rug.image}
                    collectionType={rug.collection_type}
                    // isOnSale={rug.collection_type === 4} // faqat 4 => sale
                  />
                ))}
              </div>

              {/* Pagination */}
              {totalPages > 1 && (
                <div className="flex justify-center items-center gap-2 mt-8">
                  <button
                    onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                    disabled={currentPage === 1}
                    className="px-3 py-2 text-gray-600 hover:bg-gray-100 rounded disabled:opacity-50"
                  >
                    ←
                  </button>

                  {[...Array(totalPages)].map((_, i) => {
                    const page = i + 1;
                    return (
                      <button
                        key={page}
                        onClick={() => setCurrentPage(page)}
                        className={`px-3 py-2 rounded ${currentPage === page ? 'bg-[#D7CCC8] text-white' : 'text-gray-600 hover:bg-gray-100'}`}
                      >
                        {page}
                      </button>
                    );
                  })}

                  <button
                    onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                    disabled={currentPage === totalPages}
                    className="px-3 py-2 text-gray-600 hover:bg-gray-100 rounded disabled:opacity-50"
                  >
                    →
                  </button>
                </div>
              )}
            </>
          ) : (
            <div className="text-center py-12">
              {/* <p className="text-gray-500">{t('common.no_products')}</p> */}
            </div>
          )}
        </div>
      </div>
      <ContactInfoSection />
    </div>
  );
};

export default Sales;
